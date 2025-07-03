import shlex
import re
import bashlex
from typing import Dict, List, Optional, Any


class CommandParser:
    def __init__(self):
        pass

    def is_complex(self, cmd_str: str) -> bool:
        """
        Uses bashlex to determine if the command string contains complex control structures.
        """
        try:
            parts = bashlex.parse(cmd_str)
            for part in parts:
                if self._has_complex_structure(part):
                    return True
            return False
        except bashlex.errors.ParsingError:
            return True  # Fail-safe: if parse fails, treat as complex

    def _has_complex_structure(self, node) -> bool:
        """
        Recursively inspects a bashlex AST node to detect control structures.
        """
        if hasattr(node, 'kind') and node.kind in {
            'if', 'for', 'while', 'until', 'case', 'function'
        }:
            return True
        if hasattr(node, 'parts'):
            for part in node.parts:
                if self._has_complex_structure(part):
                    return True
        if hasattr(node, 'list'):
            for sub in node.list:
                if self._has_complex_structure(sub):
                    return True
        return False

    def extract_env_and_cmd(self, tokens: List[str]) -> (List[str], List[str]):
        env_vars = []
        rest = []
        for i, token in enumerate(tokens):
            if '=' in token and not token.startswith('-') and re.match(r'^\w+=', token):
                env_vars.append(token)
            else:
                rest = tokens[i:]
                break
        return env_vars, rest

    def parse_bash_command(self, tokens: List[str]) -> List[Dict[str, Any]]:
        if not tokens:
            return []

        env_vars, cmd_tokens = self.extract_env_and_cmd(tokens)
        result = []

        if env_vars:
            result.append({"command": "set_env", "args": env_vars})

        if not cmd_tokens:
            return result

        command = cmd_tokens[0]
        args = []
        flags = {}
        i = 1

        while i < len(cmd_tokens):
            token = cmd_tokens[i]
            if token.startswith('--'):
                if '=' in token:
                    key, value = token[2:].split('=', 1)
                    flags[key] = value
                else:
                    key = token[2:]
                    value = True
                    if i + 1 < len(cmd_tokens) and not cmd_tokens[i + 1].startswith('-'):
                        value = cmd_tokens[i + 1]
                        i += 1
                    flags[key] = value
            elif token.startswith('-') and len(token) > 1:
                if len(token) > 2:
                    for char in token[1:]:
                        flags[char] = True
                else:
                    key = token[1:]
                    value = True
                    if i + 1 < len(cmd_tokens) and not cmd_tokens[i + 1].startswith('-'):
                        value = cmd_tokens[i + 1]
                        i += 1
                    flags[key] = value
            else:
                args.append(token)
            i += 1

        result.append({
            "command": command,
            "args": args,
            "flags": flags
        })
        return result

    def parse(self, bash_string: str) -> List[Dict[str, Any]]:
        """
        Parse a raw bash string with multiple commands separated by &&, ||, or ;.
        Detect complex blocks and wrap them.
        """
        if self.is_complex(bash_string):
            return [{"command": "complex_command", "args": [bash_string.strip()]}]

        sequential_cmds = re.split(r'\s*(?:&&|\|\||;)\s*', bash_string.strip())

        results = []
        for cmd in sequential_cmds:
            if not cmd:
                continue
            try:
                tokens = shlex.split(cmd)
            except ValueError:
                continue
            parsed = self.parse_bash_command(tokens)
            results.extend(parsed)
        return results


if __name__ == "__main__":
    parser = CommandParser()

    commands = [
        "cd /home/user",
        "grep --color=auto 'pattern' file.txt",
        "rm -rf /tmp/*",
        "echo 'Hello, World!' > /testbed/reproduce_error.py",
        "python3 script.py --input=data.txt --verbose",
        "cd /project && python3 run.py",
        "PYTHONPATH=/testbed",
        "PYTHONPATH=/testbed python3 main.py",
        "cd /workspace/django__django__3.0 && find . -name \"*.py\" | grep -i test | head -5",
        "cd /workspace/django__django__3.0 && grep -r \"class Avg\" --include=\"*.py\" .",
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
