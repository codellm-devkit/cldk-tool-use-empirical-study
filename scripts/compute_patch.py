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
        current_file = None
        current_status = None
        in_hunk = False

        for line in self.patch:
            if line.startswith("diff --git "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    a_path = parts[2][2:]  # strip 'a/'
                    b_path = parts[3][2:]  # strip 'b/'

                    # Canonical file path → use b_path unless it’s a deletion
                    current_file = b_path if b_path != "/dev/null" else a_path

                    # Exclude "reproduce" files
                    if re.search(r'reproduce', current_file, re.IGNORECASE):
                        current_file = None
                        in_hunk = False
                        continue

                    self.files[current_file] = {
                        'status': 'modified',  # default
                        'hunks': 0,
                        'added': [],
                        'removed': []
                    }
                    in_hunk = False

            elif line.startswith("--- "):
                if current_file and line.strip() == "--- /dev/null":
                    self.files[current_file]['status'] = 'added'

            elif line.startswith("+++ "):
                if current_file and line.strip() == "+++ /dev/null":
                    self.files[current_file]['status'] = 'deleted'

            elif line.startswith("@@"):
                if current_file:
                    self.files[current_file]['hunks'] += 1
                    in_hunk = True

            elif in_hunk and current_file:
                if line.startswith('+') and not line.startswith('+++'):
                    self.files[current_file]['added'].append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    self.files[current_file]['removed'].append(line[1:])

    def _count_abc_tree_sitter(self, code: str):
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

    def _compute_abc(self, code: str) -> tuple[int, int, int]:
        try:
            return self._count_abc_tree_sitter(code)
        except Exception as e:
            print("Tree-sitter parse error:", e)
            return 0, 0, 0

    def aggregate(self) -> Dict[str, any]:
        total_files = len(self.files)
        total_hunks = sum(d['hunks'] for d in self.files.values())
        total_added_lines = sum(len(d['added']) for d in self.files.values())
        total_removed_lines = sum(len(d['removed']) for d in self.files.values())

        total_A = total_B = total_C = 0
        for fdata in self.files.values():
            added_code = "\n".join(fdata["added"])
            removed_code = "\n".join(fdata["removed"])

            A1, B1, C1 = self._compute_abc(added_code)
            A2, B2, C2 = self._compute_abc(removed_code)

            total_A += A1 + A2
            total_B += B1 + B2
            total_C += C1 + C2

        abc_mag = round(math.sqrt(total_A**2 + total_B**2 + total_C**2), 1)

        score = (
            total_files * 2 +
            total_hunks * 1 +
            (total_added_lines + total_removed_lines) / 20 +
            abc_mag / 3
        )

        if score < 5:
            difficulty = "easy"
        elif score < 30:
            difficulty = "medium"
        elif score < 50:
            difficulty = "hard"
        else:
            difficulty = "very hard"

        return {
            "file_count": total_files,
            "hunk_count": total_hunks,
            "lines_added": total_added_lines,
            "lines_removed": total_removed_lines,
            "Assignment": total_A,
            "Branch": total_B,
            "Conditional": total_C,
            "ABC_magnitude_sum": abc_mag,
            "difficulty_score": round(score, 2),
            "patch_difficulty": difficulty
        }


if __name__ == "__main__":
    # Golden Patch Metrics
    output = "golden_patch_metrics.jsonl"
    sbv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    with open(output, "w", encoding="utf-8") as fout:
        for item in sbv:
            metrics = PatchAnalyzer(item.get("patch", "")).aggregate()
            record = {"instance_id": item["instance_id"], **metrics}
            fout.write(json.dumps(record) + "\n")

    print(f"Saved metrics for {len(sbv)} instances → {output}")

    # OpenHands Patch Metrics
    oh_trajs = "../OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/output.jsonl" # path to OpenHands trajectories
    oh_output = oh_trajs.replace("output.jsonl", "patch_metrics.jsonl")
    with open(oh_output, "w", encoding="utf-8") as fout:
        with open(oh_trajs, "r", encoding="utf-8") as fin:
            for line in fin:
                item = json.loads(line)
                if "test_result" not in item or not item["test_result"]:
                    continue
                metrics = PatchAnalyzer(item["test_result"]["git_patch"]).aggregate()
                record = {"instance_id": item["instance_id"], **metrics}
                fout.write(json.dumps(record) + "\n")
    print(f"Saved metrics for OpenHands instances → {oh_output}")

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

#     test_patch_3 = "diff --git a/lib/matplotlib/patches.py b/lib/matplotlib/patches.py\nindex e062249589..0c893aac3a 100644\n--- a/lib/matplotlib/patches.py\n+++ b/lib/matplotlib/patches.py\n@@ -586,10 +586,7 @@ class Patch(artist.Artist):\n         # docstring inherited\n         if not self.get_visible():\n             return\n-        # Patch has traditionally ignored the dashoffset.\n-        with cbook._setattr_cm(\n-                 self, _dash_pattern=(0, self._dash_pattern[1])), \\\n-             self._bind_draw_path_function(renderer) as draw_path:\n+        with self._bind_draw_path_function(renderer) as draw_path:\n             path = self.get_path()\n             transform = self.get_transform()\n             tpath = transform.transform_path_non_affine(path)\ndiff --git a/reproduce_bug.py b/reproduce_bug.py\nnew file mode 100644\nindex 0000000000..2b5ef207b6\n--- /dev/null\n+++ b/reproduce_bug.py\n@@ -0,0 +1,10 @@\n+import matplotlib.pyplot as plt\n+import matplotlib as mpl\n+\n+plt.figure(figsize=(10,10))\n+ax = plt.gca()\n+ax.add_patch(mpl.patches.Rectangle((0.5,0.5),1,1, alpha=0.5, edgecolor = 'r', linewidth=4, ls=(0,(10,10))))\n+ax.add_patch(mpl.patches.Rectangle((0.5,0.5),1,1, alpha=0.5, edgecolor = 'b', linewidth=4, ls=(10,(10,10))))\n+plt.ylim([0,2])\n+plt.xlim([0,2])\n+plt.show()\n"

#     for i, patch in enumerate([test_patch_1, test_patch_2, test_patch_3], start=1):
#         metrics = PatchAnalyzer(patch).aggregate()
#         print(f"\nPatch {i} Metrics:")
#         print(json.dumps(metrics, indent=2))
