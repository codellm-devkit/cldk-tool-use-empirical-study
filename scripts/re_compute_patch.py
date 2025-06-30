import re
import math
import json
from typing import List, Dict, Any

# ------------------- Regex Definitions -------------------

# 1. ASSIGN_RE: assignment, compound assignment, increment/decrement
ASSIGN_RE = re.compile(r"""
    (?P<assign>
        ^\s*                                  # line start with optional whitespace
        [\w\[\]()., ]+                        # potential LHS
        \s*                                   # optional space
        (?:=|[\+\-\*/%&|^]=|>>=|<<=)          # assignment or compound assignment
        (?![=])                               # not followed by another '='
    )
""", re.VERBOSE)

# 2. BRANCH_RE: function/method/class calls and instantiations
BRANCH_RE = re.compile(r"""
    (?P<branch>
        \b(?:[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\(   # function or method calls
        | \bawait\b
        | \bnew\b | \bdelete\b
    )
""", re.VERBOSE)

CONTROL_FLOW_KEYWORDS = {
    "if", "for", "while", "elif", "except", "with",
    "assert", "return", "yield", "try", "finally",
    "case", "default", "match"
}

# 3. CONDITION_RE: conditionals, logic, and control flow
CONDITION_RE = re.compile(r"""
    (?P<cond>
        == | != | <= | >= | < | >                # comparison
        | \band\b | \bor\b | \bnot\b             # boolean ops
        | \bif\b | \belif\b | \belse\b           # conditionals
        | \bcase\b | \bdefault\b | \bmatch\b     # match-case
        | \btry\b | \bexcept\b | \bfinally\b     # exception handling
        | \?                                     # ternary
        | (?<!\w)!(?![!=])                        # standalone '!' not part of '!='
    )
""", re.VERBOSE)


# ------------------- ABC Analyzer -------------------

class PatchAnalyzer:
    def __init__(self, patch_text: str):
        self.lines = patch_text.splitlines()
        self.files = {}
        self._parse_patch()
        self._compute_abc()

    def _parse_patch(self):
        current = None
        in_hunk = False
        for line in self.lines:
            if line.startswith('--- '):
                current = line[4:].split('\t')[0]
                self.files.setdefault(current, {'hunks': 0, 'added': [], 'removed': []})
                in_hunk = False
            elif line.startswith('@@ '):
                self.files[current]['hunks'] += 1
                in_hunk = True
            elif in_hunk and current:
                if line.startswith('+') and not line.startswith('+++'):
                    self.files[current]['added'].append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    self.files[current]['removed'].append(line[1:])

    def _compute_abc(self):
        for fname, stats in self.files.items():
            added = '\n'.join(stats['added'])
            removed = '\n'.join(stats['removed'])

            A = len(ASSIGN_RE.findall(added)) + len(ASSIGN_RE.findall(removed))

            # Filter branches not caused by control flow
            all_branch_matches = BRANCH_RE.findall(added) + BRANCH_RE.findall(removed)
            B = 0
            for match in all_branch_matches:
                leading = match.strip().split('.')[0].split('(')[0]
                if leading not in CONTROL_FLOW_KEYWORDS:
                    B += 1

            C = len(CONDITION_RE.findall(added)) + len(CONDITION_RE.findall(removed))

            mag = round(math.sqrt(A * A + B * B + C * C), 1)
            stats.update({'A': A, 'B': B, 'C': C, 'ABC_magnitude': mag})

    def aggregate(self) -> Dict[str, Any]:
        total_files = len(self.files)
        total_hunks = sum(f['hunks'] for f in self.files.values())
        total_added = sum(len(f['added']) for f in self.files.values())
        total_removed = sum(len(f['removed']) for f in self.files.values())
        total_A = sum(f['A'] for f in self.files.values())
        total_B = sum(f['B'] for f in self.files.values())
        total_C = sum(f['C'] for f in self.files.values())
        total_mag = round(sum(f['ABC_magnitude'] for f in self.files.values()), 1)

        score = (
            total_files * 2 +
            total_hunks * 1 +
            (total_added + total_removed) / 20 +
            total_mag / 3 +
            0.0  # deterministic for reproducibility
        )

        if score < 5:
            diff = 'easy'
        elif score < 10:
            diff = 'medium'
        elif score < 20:
            diff = 'hard'
        else:
            diff = 'very hard'

        return {
            'file_count': total_files,
            'hunk_count': total_hunks,
            'lines_added': total_added,
            'lines_removed': total_removed,
            'Assignment': total_A,
            'Branch': total_B,
            'Conditional': total_C,
            'ABC_magnitude_sum': total_mag,
            'difficulty_score': round(score, 2),
            'difficulty': diff
        }


# ------------------- Sample Test -------------------

def run_test_cases():
    test_patch_1 = """\
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1124,7 +1124,10 @@ def check_related_objects(self, field, value, opts):
- if not getattr(expression, 'filterable', True):
+ hasattr(expression, 'resolve_expression') and
+ not getattr(expression, 'filterable', True)
+ ):
"""

    test_patch_2 = """\
diff --git a/bar.py b/bar.py
--- a/bar.py
+++ b/bar.py
@@ -5,6 +5,10 @@ def bar(a):
     print(a)
-    if a > 0:
-        return True
+    if a > 0 and a % 2 == 0:
+        return True
+    elif a < 0:
+        return False
+    else:
+        return None
"""

    test_patch_3 = """\
diff --git a/baz.py b/baz.py
--- a/baz.py
+++ b/baz.py
@@ -10,3 +10,5 @@ class Baz:
     pass
+
+    def new_method(self):
+        result = compute(self.value)
"""

    for i, patch in enumerate([test_patch_1, test_patch_2, test_patch_3], start=1):
        metrics = PatchAnalyzer(patch).aggregate()
        print(f"\nPatch {i} Metrics:")
        print(json.dumps(metrics, indent=2))


run_test_cases()

# if __name__ == "__main__":
    # output = sys.argv[1] if len(sys.argv) > 1 else "golden_patch_metrics.jsonl"
    # sbv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # with open(output, "w", encoding="utf-8") as fout:
    #     for item in sbv:
    #         metrics = PatchAnalyzer(item.get("patch", "")).aggregate()
    #         record = {"instance_id": item["instance_id"], **metrics}
    #         fout.write(json.dumps(record) + "\n")

    # print(f"Saved metrics for {len(sbv)} instances → {output}")