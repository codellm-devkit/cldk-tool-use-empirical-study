import re
import os
import math
import json
import subprocess
from datasets import load_dataset
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from typing import Dict, List
import getpass

# ---------------------- Build Tree-sitter Language ----------------------
LANGUAGE: Language = Language(tspython.language())
PARSER: Parser = Parser(LANGUAGE)

# ---------------------- Patch Analyzer ----------------------
class PatchAnalyzer:
    def __init__(self, patch_text: str):
        self.patch = patch_text.splitlines()
        self.files = {}
        self._parse_patch()

    def _parse_patch(self):
        current = None
        in_hunk = False
        for line in self.patch:
            if line.startswith('--- '):
                current = line[4:].strip()
                self.files.setdefault(current, {'hunks': 0, 'added': [], 'removed': []})
            elif line.startswith('@@'):
                self.files[current]['hunks'] += 1
                in_hunk = True
            elif in_hunk and current:
                if line.startswith('+') and not line.startswith('+++'):
                    self.files[current]['added'].append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    self.files[current]['removed'].append(line[1:])

    def _count_abc_tree_sitter(self, code: str):
        """Count ABC using tree-sitter AST."""
        tree = PARSER.parse(bytes(code, "utf8"))
        root = tree.root_node

        A = B = C = 0

        def traverse(node):
            nonlocal A, B, C
            if node.type in {"assignment", "augmented_assignment", 'ann_assign'}:
                A += 1
            elif node.type in {"call", "await_expression", "function_definition", "class_definition"}:
                B += 1
            elif node.type in {
                "if_statement", "elif_clause", "else_clause", "while_statement",
                "for_statement", "for_in_clause", "assert_statement", "match_statement",
                "try_statement", "except_clause", "finally_clause",
                "comparison_operator", "boolean_operator", "logical_or_operator",
                "logical_and_operator", "not_operator", "conditional_expression"
            }:
                C += 1


            for child in node.children:
                traverse(child)

        traverse(root)
        return A, B, C

    def _compute_abc(self, code: str) -> Dict[str, int]:
        try:
            return self._count_abc_tree_sitter(code)
        except Exception as e:
            print("Tree-sitter parse error:", e)
            return 0, 0, 0

    def aggregate(self) -> Dict[str, any]:
        total_files = len(self.files)
        total_hunks = sum(f['hunks'] for f in self.files.values())
        total_added = sum(len(f['added']) for f in self.files.values())
        total_removed = sum(len(f['removed']) for f in self.files.values())

        total_A = total_B = total_C = 0
        for f in self.files.values():
            added_code = "\n".join(f["added"])
            removed_code = "\n".join(f["removed"])

            A1, B1, C1 = self._compute_abc(added_code)
            A2, B2, C2 = self._compute_abc(removed_code)

            total_A += A1 + A2
            total_B += B1 + B2
            total_C += C1 + C2

        abc_mag = round(math.sqrt(total_A**2 + total_B**2 + total_C**2), 1)
        
        # Calculate difficulty score
        score = (
            total_files * 2 +
            total_hunks * 1 +
            (total_added + total_removed) / 20 +
            abc_mag / 3
        )

        if score < 5:
            diff = "easy"
        elif score < 30:
            diff = "medium"
        elif score < 50:
            diff = "hard"
        else:
            diff = "very hard"

        return {
            "file_count": total_files,
            "hunk_count": total_hunks,
            "lines_added": total_added,
            "lines_removed": total_removed,
            "Assignment": total_A,
            "Branch": total_B,
            "Conditional": total_C,
            "ABC_magnitude_sum": abc_mag,
            "difficulty_score": round(score, 2),
            "patch_difficulty": diff
        }


if __name__ == "__main__":
    # # Golden Patch Metrics
    # output = "golden_patch_metrics.jsonl"
    # sbv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # with open(output, "w", encoding="utf-8") as fout:
    #     for item in sbv:
    #         metrics = PatchAnalyzer(item.get("patch", "")).aggregate()
    #         record = {"instance_id": item["instance_id"], **metrics}
    #         fout.write(json.dumps(record) + "\n")

    # print(f"Saved metrics for {len(sbv)} instances → {output}")

    # # OpenHands Patch Metrics
    # oh_trajs = "../OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/output.jsonl" # path to OpenHands trajectories
    # oh_output = oh_trajs.replace("output.jsonl", "patch_metrics.jsonl")
    # with open(oh_output, "w", encoding="utf-8") as fout:
    #     with open(oh_trajs, "r", encoding="utf-8") as fin:
    #         for line in fin:
    #             item = json.loads(line)
    #             if "test_result" not in item or not item["test_result"]:
    #                 continue
    #             metrics = PatchAnalyzer(item["test_result"]["git_patch"]).aggregate()
    #             record = {"instance_id": item["instance_id"], **metrics}
    #             fout.write(json.dumps(record) + "\n")
    # print(f"Saved metrics for OpenHands instances → {oh_output}")

    # SWE-Agent Patch Metrics
    user = getpass.getuser()
    sa_trajs = f"../SWE-agent/trajectories/{user}/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/preds.json" # path to SWE-Agent trajectories
    sa_output = sa_trajs.replace("preds.json", "patch_metrics.jsonl")
    with open(sa_output, "w", encoding="utf-8") as fout:
        with open(sa_trajs, "r", encoding="utf-8") as fin:
            data = json.load(fin)
            for instance_id, instance in data.items():
                model_patch = instance.get("model_patch")
                if not model_patch:
                    continue
                metrics = PatchAnalyzer(model_patch).aggregate()
                record = {"instance_id": instance.get("instance_id", instance_id), **metrics}
                fout.write(json.dumps(record) + "\n")
    print(f"Saved metrics for SWE-Agent instances → {sa_output}")

#     # sample test cases
#     test_patch_1 = """\
# diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
# --- a/django/db/models/sql/query.py
# +++ b/django/db/models/sql/query.py
# @@ -1124,7 +1124,10 @@ def check_related_objects(self, field, value, opts):
# - if not getattr(expression, 'filterable', True):
# + hasattr(expression, 'resolve_expression') and
# + not getattr(expression, 'filterable', True)
# + ):
# """

#     test_patch_2 = """\
# diff --git a/bar.py b/bar.py
# --- a/bar.py
# +++ b/bar.py
# @@ -5,6 +5,10 @@ def bar(a):
#      print(a)
# -    if a > 0:
# -        return True
# +    if a > 0 and a % 2 == 0:
# +        return True
# +    elif a < 0:
# +        return False
# +    else:
# +        return None
# """

#     test_patch_3 = """\
# diff --git a/baz.py b/baz.py
# --- a/baz.py
# +++ b/baz.py
# @@ -10,3 +10,5 @@ class Baz:
#      pass
# -    result = 'negative and even' if a < 0
# +    def new_method(self):
# +        result = compute(self.value)
# """

#     for i, patch in enumerate([test_patch_1, test_patch_2, test_patch_3], start=1):
#         metrics = PatchAnalyzer(patch).aggregate()
#         print(f"\nPatch {i} Metrics:")
#         print(json.dumps(metrics, indent=2))
