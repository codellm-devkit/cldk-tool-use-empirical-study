import shlex
from typing import Dict, List, Optional, Union, Any

class CommandParser:
    def __init__(self):
        pass  # No tool_map needed for simple bash parsing

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

    def parse(self, bash_string: str) -> List[Dict[str, Any]]:
        """
        Parse a raw bash string with multiple commands separated by && ; |
        """
        # Use shlex to split while preserving quoted strings
        lexer = shlex.shlex(bash_string, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ''

        tokens = list(lexer)

        results = []
        current_tokens = []

        # Treat '&&', ';', and '|' as command separators
        separators = {'&&', ';', '|'}

        for token in tokens:
            if token in separators:
                if current_tokens:
                    parsed = self.parse_bash_command(current_tokens)
                    if parsed:
                        results.append(parsed)
                    current_tokens = []
            else:
                current_tokens.append(token)

        # Don't forget the last one
        if current_tokens:
            parsed = self.parse_bash_command(current_tokens)
            if parsed:
                results.append(parsed)

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
        "cd /workspace/django__django__3.0 && grep -r \"class Avg\" --include=\"*.py\" ."
    ]

    for cmd in commands:
        result = parser.parse(cmd)
        print(f"\n>>> {cmd}")
        for r in result:
            print(r)