from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal
import subprocess
import sys

from zss import Node, simple_distance

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# ---------------------- Build Tree-sitter Language ----------------------
LANGUAGE: Language = Language(tspython.language())
PARSER: Parser = Parser(LANGUAGE)


class CSTNode:
    def __init__(self, node: TSNode, source_code: bytes):
        self.node = node
        self.code = source_code

    def text(self):
        return self.code[self.node.start_byte:self.node.end_byte].decode()

    def children(self):
        return [CSTNode(c, self.code) for c in self.node.children]

    def label(self):
        return self.node.type

    def byte_range(self):
        return self.node.start_byte, self.node.end_byte

    def find_all_matching_subtrees(self, query: 'CSTNode', strategy: str = "exact") -> list[CSTNode]:
        matches = []

        def dfs(n: 'CSTNode'):
            if strategy == "exact":
                if n.label() == query.label() and n.text().strip() == query.text().strip():
                    matches.append(n)
            elif strategy == "similar":
                score = CSTRewriter.compute_similarity_zss(n, query)
                if score > 0.9:
                    matches.append(n)
            elif strategy == "rule":
                if n.label() == query.label() and len(n.children()) == len(query.children()):
                    matches.append(n)
            for c in n.children():
                dfs(c)

        dfs(self)
        return matches


class CSTRewriter:
    @staticmethod
    def normalize_code(code: str) -> str:
        return code.strip()

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
    def structured_replace(full_code: str, root: CSTNode, matches: list[CSTNode], replacement_node: CSTNode) -> str:
        def walk(node: CSTNode):
            if any(node.node == match.node for match in matches):
                return replacement_node.text()
            if not node.children():
                return node.text()
            return ''.join(walk(child) for child in node.children())

        return walk(root)

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

        matches = full_cst.find_all_matching_subtrees(snippet_cst, strategy=match_strategy)
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
            targets = matches if replace_all else [matches[0]]

            new_code = cls.structured_replace(full_code, full_cst, targets, replacement_cst)

            tmp_path = path.with_suffix(".tmp.py")
            tmp_path.write_text(new_code)

            if not cls.verify_code(new_code, filename=str(tmp_path)):
                tmp_path.unlink(missing_ok=True)
                sys.exit(55)

            path.write_text(new_code)
            tmp_path.unlink(missing_ok=True)
            return new_code
