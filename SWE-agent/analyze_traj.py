import os
import json
from collections import Counter, defaultdict

PATH_TO_TRAJ = "trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test"

def analyze_trajectory_actions(root_path):
    action_counter = Counter()
    actions_per_instance = defaultdict(list)
    view_proportions = []
    total_valid = 0
    total = 0
    submit = 0
    exit_submit = 0

    for subdir, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".traj"):
                traj_path = os.path.join(subdir, file)
                total_valid += 1
                try:
                    with open(traj_path, 'r') as f:
                        data = json.load(f)
                        instance_id = data.get("environment", "")
                        trajectory = data.get("trajectory", [])
                        actions = []
                        for step in trajectory:
                            action = step.get("action", "").strip()
                            if action:
                                actions.append(action)
                                action_counter[action] += 1
                        if actions:
                            actions_per_instance[instance_id].extend(actions)
                            # Count 'view' actions
                            view_count = sum(1 for act in actions if act.startswith("view") or " view" in act)
                            proportion = view_count / len(actions)
                            view_proportions.append(proportion)
                        info = data.get("info", {})
                        if info:
                            exit_status = info.get("exit_status", "")
                            if exit_status == "submitted":
                                submit += 1
                            elif "submitted" in exit_status:
                                exit_submit += 1
                                print(f"Exit submit for {instance_id}: {exit_status}")
                            else:
                                print(f"No submit for {instance_id}: {exit_status}")
                except Exception as e:
                    print(f"Error reading {traj_path}: {e}")
            elif file.endswith(".debug.log"):
                total += 1

    print(f"\nTotal instances with a trajectory: {total_valid}")
    print(f"Total instances: {total}")
    print(f"Submit count: {submit}\nExit submit count: {exit_submit}\nNo submit count: {total - submit - exit_submit}")
    if actions_per_instance:
        avg_actions = sum(len(actions) for actions in actions_per_instance.values()) / len(actions_per_instance)
        print(f"Average number of actions per instance: {avg_actions:.2f}")
    if view_proportions:
        avg_view_proportion = sum(view_proportions) / len(view_proportions)
        print(f"Average proportion of 'view' actions per instance: {avg_view_proportion:.2%}")

    return action_counter, actions_per_instance

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Analyze SWE-agent trajectory actions.")
    # parser.add_argument("path", type=str, help="Path to the directory containing .traj files.")
    # args = parser.parse_args()

    # analyze_trajectory_actions(args.path)
    analyze_trajectory_actions(PATH_TO_TRAJ)
