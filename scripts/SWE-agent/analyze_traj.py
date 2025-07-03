import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from commandParser import CommandParser

PATH_TO_TRAJ = "../../SWE-agent/trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test"

def plot_turn_chart(dataframe, title, filename):
    fig, ax = plt.subplots(figsize=(20, 10))
    dataframe.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel("Turn Index", fontsize=14)
    ax.set_ylabel("Action Count", fontsize=14)
    ax.set_title(title, fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=12)

    legend = ax.legend(title="Action Type", ncol=2, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    legend.get_frame().set_linewidth(0.0)

    plt.tight_layout()
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(f"analysis/{filename}", dpi=300)
    plt.close()

def normalize_action(parser: CommandParser, action_str):
    parsed_cmds = parser.parse(action_str)
    if not parsed_cmds:
        return ["<unrecognized>"]

    normalized = []
    for cmd in parsed_cmds:
        if "tool" in cmd:
            label = f"{cmd['tool']}:{cmd.get('subcommand', '')}".strip(':')
        else:
            label = cmd.get("command", "<unknown>")
        normalized.append(label)
    return normalized

def analyze_trajectory_actions(root_path):
    parser = CommandParser()
    parser.load_tool_yaml_files([
        "../../SWE-agent/tools/edit_anthropic/config.yaml",
        "../../SWE-agent/tools/review_on_submit_m/config.yaml",
        "../../SWE-agent/tools/registry/config.yaml"
    ])

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
                            action_str = step.get("action", "").strip()
                            if action_str:
                                normalized_actions = normalize_action(parser, action_str)
                                actions.extend(normalized_actions)
                                for norm in normalized_actions:
                                    action_counter[norm] += 1
                                    turn_action_counts[i][norm] += 1
                        if actions:
                            actions_per_instance[instance_id].extend(actions)
                            view_count = sum(1 for act in actions if "view" in act)
                            proportion = view_count / len(actions)
                            view_proportions.append(proportion)
                        info = data.get("info", {})
                        if info:
                            exit_status = info.get("exit_status", "")
                            if exit_status == "submitted":
                                submit += 1
                            elif "submitted" in exit_status:
                                exit_submit += 1
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

    df = pd.DataFrame(turn_action_counts).fillna(0).astype(int).T
    plot_turn_chart(df, "Action Type Frequency at Each Turn", "turn_action_distribution.jpg")
    df_first50 = df[df.index < 50]
    plot_turn_chart(df_first50, "Action Frequency at First 50 Turns", "turn_action_distribution_first50.jpg")

    return action_counter, actions_per_instance

if __name__ == "__main__":
    analyze_trajectory_actions(PATH_TO_TRAJ)
