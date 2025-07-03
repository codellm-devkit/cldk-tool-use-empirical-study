import yaml
import shlex
import re
import bashlex
from typing import Dict, List, Optional, Any


class ToolDefinition:
    def __init__(self, tool_name: str, subcommands: List[str], arg_specs: Dict[str, dict]):
        self.tool_name = tool_name
        self.subcommands = subcommands
        self.arg_specs = arg_specs

    def parse(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        if not tokens or tokens[0] != self.tool_name:
            return None

        has_subcommand = bool(self.subcommands)
        subcommand = tokens[1] if has_subcommand else None
        if has_subcommand and subcommand not in self.subcommands:
            return None

        parsed = {
            "tool": self.tool_name,
            "subcommand": subcommand,
            "args": {}
        }

        args = tokens[2:] if has_subcommand else tokens[1:]
        positional = [
            spec for spec in self.arg_specs.values()
            if "argument_format" not in spec and spec.get("name") != "command"
        ]

        i = 0
        pos_idx = 0
        while i < len(args):
            token = args[i]
            if token.startswith("--"):
                key = token[2:]
                spec = self.arg_specs.get(key)
                if not spec:
                    i += 1
                    continue

                value = True
                arg_type = spec.get("type")

                if arg_type == "array":
                    i += 1
                    value = []
                    while i < len(args) and not args[i].startswith("--"):
                        try:
                            value.append(int(args[i]))
                        except ValueError:
                            break
                        i += 1
                    parsed["args"][key] = value
                    continue
                elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                    value = args[i + 1]
                    if arg_type == "integer":
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    parsed["args"][key] = value
                    i += 2
                    continue
                else:
                    parsed["args"][key] = value
            else:
                if pos_idx < len(positional):
                    name = positional[pos_idx]["name"]
                    value = token
                    if positional[pos_idx].get("type") == "integer":
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    parsed["args"][name] = value
                    pos_idx += 1
            i += 1

        return parsed


class CommandParser:
    def __init__(self):
        self.tool_map: Dict[str, ToolDefinition] = {}

    def load_tool_yaml_files(self, yaml_paths: List[str]):
        for path in yaml_paths:
            try:
                with open(path, "r") as f:
                    content = yaml.safe_load(f)
                    for tool_name, spec in content.get("tools", {}).items():
                        if not spec:
                            continue
                        subcommands = next(
                            (arg.get("enum", []) for arg in spec.get("arguments", []) if arg.get("name") == "command"),
                            []
                        )
                        arg_specs = {arg["name"]: arg for arg in spec.get("arguments", [])}
                        self.tool_map[tool_name] = ToolDefinition(tool_name, subcommands, arg_specs)
            except Exception as e:
                print(f"Failed to load YAML from {path}: {e}")

    def is_complex(self, cmd_str: str) -> bool:
        try:
            parts = bashlex.parse(cmd_str)
            for part in parts:
                if self._has_control_structure(part):
                    return True
            return False
        except bashlex.errors.ParsingError:
            return True  # treat parsing errors as complex

    def _has_control_structure(self, node) -> bool:
        if hasattr(node, 'kind') and node.kind in {
            'if', 'for', 'while', 'until', 'case', 'function'
        }:
            return True
        if hasattr(node, 'parts'):
            for part in node.parts:
                if self._has_control_structure(part):
                    return True
        if hasattr(node, 'list'):
            for sub in node.list:
                if self._has_control_structure(sub):
                    return True
        return False

    def split_env_and_command(self, cmd_str: str) -> List[str]:
        env_pattern = re.compile(r"^((?:\w+=[^ \t\n\r\f\v]+[ \t]*)+)(.+)?")
        match = env_pattern.match(cmd_str.strip())

        def split_sequential(cmd: str) -> List[str]:
            return [c for c in re.split(r'\s*(?:&&|\|\||;)\s*', cmd.strip()) if c]

        if match:
            env_part = match.group(1).strip()
            rest = match.group(2) or ""
            return [env_part] + split_sequential(rest)
        else:
            return split_sequential(cmd_str)

    def parse(self, cmd_str: str) -> List[Dict[str, Any]]:
        if self.is_complex(cmd_str):
            return [{"command": "complex_command", "args": [cmd_str.strip()]}]

        parts = self.split_env_and_command(cmd_str)
        results = []

        for subcmd in parts:
            if re.fullmatch(r"\w+=.+", subcmd.strip()) or all("=" in token for token in subcmd.strip().split()):
                results.append({"command": "set_env", "args": [subcmd.strip()]})
                continue

            try:
                tokens = shlex.split(subcmd.strip())
            except ValueError:
                continue

            if not tokens:
                continue

            tool = tokens[0]
            if tool in self.tool_map:
                result = self.tool_map[tool].parse(tokens)
                if result:
                    results.append(result)
            else:
                result = self.parse_bash_command(tokens)
                if result:
                    results.append(result)
        return results

    def parse_bash_command(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        if not tokens:
            return None

        command = tokens[0]
        args = []
        flags = {}
        i = 1

        while i < len(tokens):
            token = tokens[i]
            if token.startswith('--'):
                if '=' in token:
                    key, value = token[2:].split('=', 1)
                    flags[key] = value
                else:
                    key = token[2:]
                    value = True
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                        value = tokens[i + 1]
                        i += 1
                    flags[key] = value
            elif token.startswith('-') and len(token) > 1:
                if len(token) > 2:
                    for char in token[1:]:
                        flags[char] = True
                else:
                    key = token[1:]
                    value = True
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                        value = tokens[i + 1]
                        i += 1
                    flags[key] = value
            else:
                args.append(token)
            i += 1

        return {
            "command": command,
            "args": args,
            "flags": flags
        }


if __name__ == "__main__":
    parser = CommandParser()
    parser.load_tool_yaml_files([
        "../../SWE-agent/tools/edit_anthropic/config.yaml",
        "../../SWE-agent/tools/review_on_submit_m/config.yaml",
        "../../SWE-agent/tools/registry/config.yaml"
    ])

    commands = [
        "str_replace_editor view /testbed/foo.py",
        "str_replace_editor view /testbed/bar.py --view_range 1 10",
        "str_replace_editor create /testbed/test.py --file_text 'text'",
        "str_replace_editor str_replace /testbed/foo.py --old_str 'some_old' --new_str 'some_new'",
        "submit",
        "cd /home/user",
        "grep --color=auto 'pattern' file.txt",
        "rm -rf /tmp/*",
        "echo 'Hello, World!' > /testbed/reproduce_error.py",
        "python3 script.py --input=data.txt --verbose",
        "cd /project && python3 run.py",
        "PYTHONPATH=/testbed",
        "PYTHONPATH=/project python3 script.py --input file.txt && echo \"done\" ; rm temp.log"
    ]

    for cmd in commands:
        result = parser.parse(cmd)
        print(f"\n>>> {cmd}")
        for r in result:
            print(r)

    complex_bash = """
for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
    if [ -f "$file" ] && (file "$file" | grep -q "executable" || git check-attr binary "$file" | grep -q "binary: set"); then
        git rm -f "$file" 2>/dev/null || rm -f "$file"
        echo "Removed: $file"
    fi
done
    """

    result = parser.parse(complex_bash)
    print(f"\n>>> Complex Bash:\n{complex_bash}")
    for r in result:
        print(r)
