import yaml
import re
from typing import Dict, List, Optional, Union
from shlex import split as shell_split


class ToolDefinition:
    def __init__(self, tool_name: str, commands: List[str], arg_specs: Dict[str, dict]):
        self.tool_name = tool_name
        self.commands = commands  # allowed subcommands
        self.arg_specs = arg_specs  # dict of arg_name -> spec

    def match_command(self, tokens: List[str]) -> Optional[Dict]:
        if not tokens or tokens[0] != self.tool_name:
            return None

        has_subcommand = bool(self.commands)
        subcommand = tokens[1] if has_subcommand else None
        if has_subcommand and subcommand not in self.commands:
            return None

        parsed = {
            "tool": self.tool_name,
            "subcommand": subcommand,
            "args": {}
        }

        # extract argument values from flags (e.g. --old_str val)
        args = tokens[2:] if has_subcommand else tokens[1:]
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                arg_name = args[i][2:]
                arg_spec = self.arg_specs.get(arg_name)
                if not arg_spec:
                    i += 1
                    continue
                arg_type = arg_spec.get("type")
                if arg_type == "array":
                    values = []
                    i += 1
                    while i < len(args) and not args[i].startswith("--"):
                        try:
                            values.append(int(args[i]))
                        except ValueError:
                            break
                        i += 1
                    parsed["args"][arg_name] = values
                elif arg_type == "integer":
                    if i + 1 < len(args):
                        try:
                            parsed["args"][arg_name] = int(args[i + 1])
                            i += 2
                        except ValueError:
                            i += 1
                    else:
                        i += 1
                else:  # string
                    if i + 1 < len(args):
                        parsed["args"][arg_name] = args[i + 1]
                        i += 2
                    else:
                        i += 1
            else:
                # Positional args
                expected = [arg for arg in self.arg_specs.values()
                            if "argument_format" not in arg and arg.get("name") != "command"]
                pos_idx = len(parsed["args"])
                if pos_idx < len(expected):
                    arg_spec = expected[pos_idx]
                    name = arg_spec["name"]
                    value = args[i]
                    if arg_spec.get("type") == "integer":
                        value = int(value)
                    parsed["args"][name] = value
                i += 1

        return parsed


def load_tool_definitions(yaml_paths: List[str]) -> Dict[str, ToolDefinition]:
    tool_map = {}
    for path in yaml_paths:
        try:
            with open(path, "r") as f:
                content = yaml.safe_load(f)
                tools = content.get("tools", {})
                for tool_name, spec in tools.items():
                    if not spec:
                        continue
                    commands = []
                    for arg in spec.get("arguments", []):
                        if arg.get("name") == "command" and "enum" in arg:
                            commands = arg["enum"]
                            break
                    arg_specs = {arg["name"]: arg for arg in spec.get("arguments", [])}
                    tool_map[tool_name] = ToolDefinition(tool_name, commands, arg_specs)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return tool_map


def parse_tool_command(cmd_str: str, tool_map: Dict[str, ToolDefinition]) -> Optional[Dict]:
    try:
        tokens = shell_split(cmd_str.strip())
    except ValueError:
        print(f"Invalid shell command: {cmd_str}")
        return None
    if not tokens:
        return None
    tool_name = tokens[0]
    tool_def = tool_map.get(tool_name)
    if tool_def:
        return tool_def.match_command(tokens)
    return None


if __name__ == "__main__":
    yaml_files = [
        "tools/edit_anthropic/config.yaml",
        "tools/review_on_submit_m/config.yaml"
    ]

    tool_map = load_tool_definitions(yaml_files)

    commands = [
        "str_replace_editor view /testbed/foo.py",
        "str_replace_editor view /testbed/bar.py --view_range 1 10",
        "str_replace_editor str_replace /testbed/foo.py --old_str 'some_old' --new_str 'some_new'",
        "submit"
    ]

    for cmd in commands:
        result = parse_tool_command(cmd, tool_map)
        print(f">>> {cmd}\nParsed: {result}\n")
