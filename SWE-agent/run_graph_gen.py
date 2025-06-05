import os
import json
from commandParser import CommandParser
from buildGraph import build_graph_from_trajectory
import sys


def load_eval_status(eval_report_path):
    with open(eval_report_path, "r") as f:
        report = json.load(f)
    resolved = set(report.get("resolved_ids", []))
    unresolved = set(report.get("unresolved_ids", []))
    return resolved, unresolved


def batch_generate_graphs(root_instance_dir, eval_report_path):
    resolved_ids, unresolved_ids = load_eval_status(eval_report_path)

    parser = CommandParser()
    parser.load_tool_yaml_files([
        "tools/edit_anthropic/config.yaml",
        "tools/review_on_submit_m/config.yaml",
        "tools/registry/config.yaml"
    ])

    idx = 0
    for instance_name in os.listdir(root_instance_dir):
        instance_dir = os.path.join(root_instance_dir, instance_name)
        if not os.path.isdir(instance_dir):
            continue

        traj_file = None
        for fname in os.listdir(instance_dir):
            if fname.endswith(".traj"):
                traj_file = os.path.join(instance_dir, fname)
                break

        if not traj_file:
            print(f"No .traj file found in {instance_dir}")
            continue

        if instance_name in resolved_ids:
            subdir = "resolved"
        elif instance_name in unresolved_ids:
            subdir = "unresolved"
        else:
            subdir = "unsubmitted"

        output_prefix = traj_file.replace("trajectories", f"graphs/{subdir}").replace(".traj", "")
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

        with open(traj_file, "r") as f:
            traj_data = json.load(f)
        idx += 1
        print(f"{idx}. {instance_name} ({subdir})...")
        build_graph_from_trajectory(traj_data, parser, output_prefix, eval_report_path)


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python batch_graph_generator.py <root_instance_dir> <eval_report_path>")
    #     sys.exit(1)

    # root_dir = sys.argv[1]
    # eval_report = sys.argv[2]
    root_dir = "trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test"
    eval_report = "sb-cli-reports/Subset.swe_bench_verified__test__evaluate_swev.json"
    batch_generate_graphs(root_dir, eval_report)
