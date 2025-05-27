import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd

PATH_TO_TRAJ = "trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test"

def normalize_action(action_str):
    """Extract command name from action string."""
    if action_str.startswith("str_replace_editor") or action_str.startswith("sudo"):
        parts = action_str.split()
        return parts[1] if len(parts) > 1 else "str_replace_editor"
    return action_str.split()[0]

def plot_turn_chart(dataframe, title, filename):
    fig, ax = plt.subplots(figsize=(20, 10))
    dataframe.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel("Turn Index", fontsize=14)
    ax.set_ylabel("Action Count", fontsize=14)
    ax.set_title(title, fontsize=16)

    # Remove chart borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Beautify ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Legend inside, two columns
    legend = ax.legend(title="Action Type", ncol=2, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    legend.get_frame().set_linewidth(0.0)  # No border

    plt.tight_layout()
    plt.savefig(f"analysis/{filename}", dpi=300)
    plt.close()

def analyze_trajectory_actions(root_path):
    action_counter = Counter()
    actions_per_instance = defaultdict(list)
    view_proportions = []
    turn_action_counts = defaultdict(Counter)

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
                        for i, step in enumerate(trajectory):
                            action = step.get("action", "").strip()
                            if action:
                                normalized = normalize_action(action)
                                actions.append(normalized)
                                action_counter[normalized] += 1
                                turn_action_counts[i][normalized] += 1
                        if actions:
                            actions_per_instance[instance_id].extend(actions)
                            view_count = sum(1 for act in actions if act == "view")
                            proportion = view_count / len(actions)
                            view_proportions.append(proportion)
                        info = data.get("info", {})
                        if info:
                            exit_status = info.get("exit_status", "")
                            if exit_status == "submitted":
                                submit += 1
                            elif "submitted" in exit_status:
                                exit_submit += 1
                            else:
                                pass  # no submit
                except Exception as e:
                    print(f"Error reading {traj_path}: {e}")
            elif file.endswith(".debug.log"):
                total += 1

    print(f"\nTotal instances with a trajectory: {total_valid}")
    print(f"Total instances: {total}")
    print(f"Submit count: {submit}\nExit submit count: {exit_submit}\nNo submit count: {total - submit - exit_submit}")
    if actions_per_instance:
        print(f"Maximum number of actions in an instance: {max(len(actions) for actions in actions_per_instance.values())}")
        print(f"Minimum number of actions in an instance: {min(len(actions) for actions in actions_per_instance.values())}")
        avg_actions = sum(len(actions) for actions in actions_per_instance.values()) / len(actions_per_instance)
        print(f"Average number of actions per instance: {avg_actions:.2f}")
    if view_proportions:
        avg_view_proportion = sum(view_proportions) / len(view_proportions)
        print(f"Average proportion of 'view' actions per instance: {avg_view_proportion:.2%}")
    
    print(f"Action types: {len(action_counter)}\n{action_counter}")

    os.makedirs("analysis", exist_ok=True)

    # Create a DataFrame for turn-action counts
    df = pd.DataFrame(turn_action_counts).fillna(0).astype(int).T

    # Plot full range of turns
    plot_turn_chart(df, "Action Type Frequency at Each Turn", "turn_action_distribution.jpg")

    # Plot first 50 turns only
    df_first50 = df[df.index < 50]
    plot_turn_chart(df_first50, "Action Frequency at First 50 Turns", "turn_action_distribution_first50.jpg")

    return action_counter, actions_per_instance

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Analyze SWE-agent trajectory actions.")
    # parser.add_argument("path", type=str, help="Path to the directory containing .traj files.")
    # args = parser.parse_args()

    # analyze_trajectory_actions(args.path)
    analyze_trajectory_actions(PATH_TO_TRAJ)
