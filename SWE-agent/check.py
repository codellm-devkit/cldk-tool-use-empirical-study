import json
import os
import sys
from commandParser import CommandParser
import csv
import difflib

def check_replace(traj_data, parser: CommandParser, csv_path="str_replace_pairs.csv"):
    trajectory = traj_data.get("trajectory", [])
    replace_pairs = []  # Store (old_str, new_str) tuples
    old_strs = []       # Store old_strs for comparison

    for idx, step in enumerate(trajectory):
        action = step.get("action", "")
        parsed = parser.parse(action)

        if parsed is None:
            continue

        if parsed.get("tool") and parsed.get("subcommand", "") == "str_replace":
            args = parsed.get("args", {})
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")
            old_strs.append(old_str)
            replace_pairs.append((old_str, new_str))

    # Compare old_strs pairwise and print diffs
    for i in range(len(old_strs) - 1):
        for j in range(i + 1, len(old_strs)):
            if old_strs[i] == old_strs[j]:
                print(f"equal: pair ({i + 1}, {j + 1})")
            else:
                print(f"not equal: pair ({i + 1}, {j + 1})")

                # Use difflib to highlight differences
                sm = difflib.SequenceMatcher(None, old_strs[i], old_strs[j])
                diff_i, diff_j = "", ""

                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag != 'equal':
                        diff_i += old_strs[i][i1:i2]
                        diff_j += old_strs[j][j1:j2]

                print(f"  → differing in: [{diff_i!r}] vs [{diff_j!r}]")

    # Write to CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["old_str", "new_str"])
        writer.writerows(replace_pairs)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python buildGraph.py path/to/instance_dir")
    #     sys.exit(1)

    # instance_dir = sys.argv[1]
    # if not os.path.isdir(instance_dir):
    #     print(f"Error: {instance_dir} is not a valid directory.")
    #     sys.exit(1)

    instance_dir = "trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/astropy__astropy-8872"
    # Find .traj file
    traj_file = None
    for fname in os.listdir(instance_dir):
        if fname.endswith(".traj"):
            traj_file = os.path.join(instance_dir, fname)
            break

    if not traj_file:
        print(f"No .traj file found in {instance_dir}")
        sys.exit(0)

    with open(traj_file, "r") as f:
        traj_data = json.load(f)

    # Generate output path: replace "trajectories" with "graphs"
    output_prefix = traj_file.replace("trajectories", "graphs").replace(".traj", "")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    parser = CommandParser()
    parser.load_tool_yaml_files([
        "tools/edit_anthropic/config.yaml",
        "tools/review_on_submit_m/config.yaml",
        "tools/registry/config.yaml"
    ])

    check_replace(traj_data, parser)
    