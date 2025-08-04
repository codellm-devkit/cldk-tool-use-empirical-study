from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal
import subprocess
import sys

from zss import Node, simple_distance
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node as TSNode

try:
    import black
except ImportError:
    black = None

LANGUAGE: Language = Language(tspython.language())
PARSER: Parser = Parser(LANGUAGE)

class CSTNode:
    """
    A wrapper around tree-sitter nodes to provide a more convenient interface for working with the concrete syntax tree (CST) of Python code.
    """
    def __init__(self, node: TSNode, source_code: bytes):
        self.node = node
        self.code = source_code

    def text(self):
        return self.code[self.node.start_byte:self.node.end_byte].decode()

    def children(self):
        return [CSTNode(c, self.code) for c in self.node.children if c.type != ","]

    def label(self):
        return self.node.type

    def byte_range(self):
        return self.node.start_byte, self.node.end_byte

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CSTNode):
            return NotImplemented
        return self.node.id == other.node.id

    def walk_tree(self):
        stack = [self.node]
        while stack:
            current = stack.pop()
            yield CSTNode(current, self.code)
            stack.extend(reversed(current.children))

    def find_all_matching_subtrees(self, query: CSTNode, strategy: str = "exact") -> list[CSTNode]:
        # Find all subtrees in the current CST that match the query node.
        if strategy not in {"exact", "similar", "rule"}:
            print(f"Invalid match strategy: {strategy}. Use 'exact', 'similar', or 'rule'.")
            sys.exit(52)
            
        matches = []
        print(f"<DEBUG> Checking query:\n{query.label()} - {query.text().strip()}")

        for candidate in self.walk_tree():
            # print(f"<DEBUG> Checking candidate:\n{candidate.label()} - {candidate.text().strip()}")

            if strategy == "exact":
                if candidate.label() == query.label() and candidate.text().strip() == query.text().strip():
                    matches.append(candidate)
            elif strategy == "similar":
                if candidate.label() == query.label():
                    score = CSTRewriter.compute_similarity_zss(candidate, query)
                    if score > 0.9:
                        matches.append(candidate)
            elif strategy == "rule":
                if candidate.label() == query.label() and len(candidate.children()) == len(query.children()):
                    matches.append(candidate)

        print(f"<DEBUG> Found {len(matches)} matches:\n" + "\n".join(f"{m.label()} - {m.text().strip()}" for m in matches))
        return matches


class CSTRewriter:
    @staticmethod
    def normalize_code(code: str, lint: bool = True) -> str:
        code = code.strip()
        if lint and black:
            try:
                code = black.format_str(code, mode=black.Mode())
            except Exception:
                pass
        return code

    @staticmethod
    def parse_to_cst(code: str) -> CSTNode:
        tree = PARSER.parse(code.encode("utf-8"))
        return CSTNode(tree.root_node, code.encode("utf-8"))

    @staticmethod
    def unparse_cst(node: CSTNode) -> str:
        return node.text()

    @staticmethod
    def compute_similarity_zss(a: CSTNode, b: CSTNode) -> float:
        class ZSSNode(Node):
            def __init__(self, cst: CSTNode):
                self._cst = cst
                super().__init__()

            def get_children(self):
                return [ZSSNode(c) for c in self._cst.children()]

            def get_label(self):
                return self._cst.label()

        dist = simple_distance(ZSSNode(a), ZSSNode(b))
        max_size = max(CSTRewriter.count_nodes(a), CSTRewriter.count_nodes(b))
        return 1.0 - (dist / max_size) if max_size > 0 else 1.0

    @staticmethod
    def count_nodes(cst: CSTNode) -> int:
        return 1 + sum(CSTRewriter.count_nodes(c) for c in cst.children())

    @staticmethod
    def structured_replace(full_code: str, root: CSTNode, targets: list[CSTNode], replacement_node: CSTNode) -> str:
        """
        Replace target CSTNodes within the root CST by regenerating the full code recursively.
        Ensures the output is structurally valid Python code.
        """
        target_ids = {t.node.id for t in targets}

        def reconstruct(node: CSTNode) -> str:
            if node.node.id in target_ids:
                return replacement_node.text()
            elif not node.node.children:
                return node.text()
            else:
                parts = []
                prev_end = node.node.start_byte
                for child in node.node.children:
                    cst_child = CSTNode(child, node.code)
                    parts.append(node.code[prev_end:child.start_byte].decode("utf-8"))  # interstitial whitespace/punct
                    parts.append(reconstruct(cst_child))
                    prev_end = child.end_byte
                parts.append(node.code[prev_end:node.node.end_byte].decode("utf-8"))  # trailing content
                return ''.join(parts)

        return reconstruct(root)


    @staticmethod
    def verify_code(code: str, filename: Optional[str] = None) -> bool:
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            print(f"<ERROR> Code verification failed: {e}")
            return False

        if filename:
            errors = CSTRewriter.run_flake8(filename)
            if errors.strip():
                print(errors)
        return True

    @staticmethod
    def run_flake8(file_path: str) -> str:
        try:
            result = subprocess.run(
                ["flake8", "--isolated", "--select=E,F", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Warning: Failed to run pre-edit linter on {file_path}: {e}"

    @classmethod
    def apply_rw(
        cls,
        file_path: str,
        snippet: str,
        mode: Literal["extract", "replace"],
        replacement: Optional[str] = None,
        match_strategy: Literal["exact", "similar", "rule"] = "exact",
        replace_all: bool = False,
    ) -> str:
        path = Path(file_path)
        if not path.exists():
            print(f"<ERROR> File not found: {file_path}")
            sys.exit(50)

        try:
            full_code = path.read_text()
        except Exception as e:
            print(f"<ERROR> Cannot read file {file_path}: {e}")
            sys.exit(51)

        full_code_norm = cls.normalize_code(full_code)
        snippet_code = cls.normalize_code(snippet)

        full_cst = cls.parse_to_cst(full_code_norm)
        snippet_cst = cls.parse_to_cst(snippet_code)

        print(f"<DEBUG> Full CST: {full_cst.node}\n")
        print(f"Full code:\n{full_cst.text()}\n")
        print(f"<DEBUG> Snippet CST: {snippet_cst.node}\n")
        print(f"Snippet code:\n{snippet_cst.text()}\n")

        snippet_body_nodes = list(snippet_cst.walk_tree())
        snippet_body = snippet_body_nodes[1] if len(snippet_body_nodes) > 1 else snippet_cst
        matches = full_cst.find_all_matching_subtrees(snippet_body, strategy=match_strategy)

        if not matches:
            print("<ERROR> No matching subtree found.")
            sys.exit(53)

        if mode == "extract":
            return '\n\n'.join(cls.unparse_cst(m) for m in matches)

        if mode == "replace":
            if not replacement:
                print("<ERROR> Replacement code required.")
                sys.exit(54)

            replacement_cst = cls.parse_to_cst(replacement)
            replacement_nodes = list(replacement_cst.walk_tree())
            replacement_body = replacement_nodes[1] if len(replacement_nodes) > 1 else replacement_cst
            targets = matches if replace_all else [matches[0]]

            new_code = cls.structured_replace(full_code, full_cst, targets, replacement_body)

            # print(f"<DEBUG> New code:\n{new_code}\n")
            tmp_path = path.with_suffix(".tmp.py")
            tmp_path.write_text(new_code)

            if not cls.verify_code(new_code, filename=str(tmp_path)):
                tmp_path.unlink(missing_ok=True)
                sys.exit(55)

            path.write_text(new_code)
            tmp_path.unlink(missing_ok=True)
            return new_code


import tempfile
from pathlib import Path
import os


def write_temp_file(content: str) -> Path:
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".py").name
    Path(path).write_text(content)
    return Path(path)

def delete_temp_file(path: Path):
    if path.exists():
        try:
            path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete temporary file {path}: {e}")

def test_single_replace():
    original = """
def greet():
    print("hello")
    print("hello")
"""
    snippet = 'print("hello")'
    replacement = 'print("hi")'

    path = write_temp_file(original)
    print(path)
    CSTRewriter.apply_rw(
        file_path=str(path),
        snippet=snippet,
        mode="replace",
        replacement=replacement,
        match_strategy="exact",
        replace_all=False,
    )

    result = path.read_text()
    print(result)
    assert result.count('print("hi")') == 1
    assert result.count('print("hello")') == 1
    delete_temp_file(path)
    print("✅ test_single_replace passed.")


def test_multi_replace():
    original = """
def greet():
    print("hello")
    print("hello")
"""
    snippet = 'print("hello")'
    replacement = 'print("hi")'

    path = write_temp_file(original)
    CSTRewriter.apply_rw(
        file_path=str(path),
        snippet=snippet,
        mode="replace",
        replacement=replacement,
        match_strategy="exact",
        replace_all=True,
    )

    result = path.read_text()
    assert result.count('print("hi")') == 2
    assert 'print("hello")' not in result
    delete_temp_file(path)
    print("✅ test_multi_replace passed.")


def test_extract_all():
    original = """
def f():
    x = 1
    x = 1
"""
    snippet = "x = "
    path = write_temp_file(original)

    extracted = CSTRewriter.apply_rw(
        file_path=str(path),
        snippet=snippet,
        mode="extract",
        match_strategy="exact",
        replace_all=True,
    )

    assert extracted.count("x = ") == 2
    delete_temp_file(path)
    print("✅ test_extract_all passed.")


if __name__ == "__main__":
    test_single_replace()
    test_multi_replace()
    test_extract_all()
