import json
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from commandParser import CommandParser
from pathlib import Path

def get_phase(tool: str, subcommand: str, command: str, label: str) -> str:
    if tool == "str_replace_editor" and subcommand == "view":
        return "localization"
    if tool == "str_replace_editor" and subcommand == "str_replace":
        return "patch"
    if command == "python" or (tool == "str_replace_editor" and subcommand == "create"):
        return "verification"
    if tool == "submit":
        return "submit"
    return "general"

def get_phase_color(phase: str) -> str:
    return {
        "localization": "lightcoral",
        "patch": "gold",
        "verification": "palegreen",
        "submit": "mediumpurple",
        "general": "skyblue"
    }.get(phase, "skyblue")

def check_edit_status(tool, subcommand, args, observation):
    if tool != "str_replace_editor" or subcommand != "str_replace" or not observation:
        return None

    if "has been edited." in observation:
        return "success"
    if "did not appear verbatim" in observation:
        return "failure: not found"
    if "Multiple occurrences of old_str" in observation:
        return "failure: multiple occurrences"
    if "old_str" in observation and "is the same as new_str" in observation:
        return "failure: no change"

    return "failure: unknown"

def determine_resolution_status(instance_path: str, eval_report_path: str) -> str:
    with open(eval_report_path, 'r') as f:
        report = json.load(f)
    
    instance_id = os.path.basename(instance_path)
    
    if instance_id in report.get("resolved_ids", []):
        return "resolved"
    elif instance_id in report.get("unresolved_ids", []):
        return "unresolved"
    return "unsubmitted"

def build_graph_from_trajectory(traj_data, parser: CommandParser, output_prefix: str, eval_report_path: str):
    G = nx.MultiDiGraph()
    trajectory = traj_data.get("trajectory", [])
    previous_node = None
    node_counter = 0
    localization_nodes = []

    for step_idx, step in enumerate(trajectory):
        action_str = step.get("action", "")
        execution_time = step.get("execution_time", 0.0)
        state = step.get("state", {})

        if action_str.strip() == "":
            node_label = "empty action"
            node_key = f"{node_counter}:{node_label}"
            G.add_node(node_key, label=node_label, execution_time=execution_time, state=state, args={}, phase="general")
            if previous_node is not None:
                G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")
            previous_node = node_key
            node_counter += 1
            continue

        parsed_commands = parser.parse(action_str)
        if not parsed_commands:
            continue

        time_per_command = execution_time / len(parsed_commands)

        for parsed in parsed_commands:
            tool = parsed.get("tool", "")
            subcommand = parsed.get("subcommand", "")
            command = parsed.get("command", "")
            args = parsed.get("args", {})

            if tool:
                node_label = f"{tool}\n{subcommand}" if subcommand else tool
            else:
                node_label = command or action_str.strip()
            phase = get_phase(tool, subcommand, command, node_label)

            edit_status = check_edit_status(tool, subcommand, args, step.get("observation", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = f"{node_counter}:{node_label}"
            G.add_node(
                node_key,
                label=node_label,
                execution_time=time_per_command,
                state=state,
                args=args,
                phase=phase
            )

            if phase == "localization":
                localization_nodes.append(node_key)

            if previous_node is not None:
                G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")

            previous_node = node_key
            node_counter += 1

    build_hierarchical_edges(G, localization_nodes)

    resolution_status = determine_resolution_status(output_prefix, eval_report_path)
    G.graph["resolution_status"] = resolution_status
    G.graph["instance_name"] = os.path.basename(output_prefix)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    with open(output_prefix + ".json", "w") as f:
        json.dump(json_graph.node_link_data(G, edges="edges"), f, indent=2)

    _draw_graph(G, output_prefix + ".png")

def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    view_nodes = []
    for node in localization_nodes:
        path = G.nodes[node].get("args").get("path")
        if path:
            view_nodes.append((node, Path(path), "view_range" in G.nodes[node].get("args", {})))

    view_nodes.sort(key=lambda x: len(x[1].parts))
    path_to_node = {str(path): node for node, path, is_range in view_nodes if not is_range}

    for node, path, is_range in view_nodes:
        if is_range:
            parent = path_to_node.get(str(path))
            if parent:
                G.add_edge(parent, node, type="hier")
        else:
            for parent_path in path.parents:
                parent_node = path_to_node.get(str(parent_path))
                if parent_node:
                    G.add_edge(parent_node, node, type="hier")
                    break

def _draw_graph(G: nx.MultiDiGraph, png_path: str):
    from matplotlib.patches import Patch
    from networkx.drawing.nx_agraph import graphviz_layout

    plt.figure(figsize=(max(12, len(G.nodes) * 0.6), 10))

    try:
        pos = graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G, seed=42)

    phase_colors = {
        "localization": "lightcoral",
        "patch": "gold",
        "verification": "palegreen",
        "submit": "mediumpurple",
        "general": "skyblue"
    }

    node_colors = []
    labels = {}
    for node, data in G.nodes(data=True):
        color = phase_colors.get(data.get("phase", "general"), "skyblue")
        status = data.get("args", {}).get("edit_status") if isinstance(data.get("args"), dict) else None

        if status == "success":
            data["label"] += " ✓"
        elif status and status.startswith("failure"):
            data["label"] += " ✗"

        labels[node] = data["label"]
        node_colors.append(color)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    exec_edges = [(u, v) for u, v, _, d in G.edges(keys=True, data=True) if d.get("type") == "exec"]
    hier_edges = [(u, v) for u, v, _, d in G.edges(keys=True, data=True) if d.get("type") == "hier"]

    nx.draw_networkx_edges(G, pos, edgelist=exec_edges, edge_color="gray", arrows=True, connectionstyle="arc3,rad=0.0")
    nx.draw_networkx_edges(G, pos, edgelist=hier_edges, edge_color="green", arrows=True, style="dashed", connectionstyle="arc3,rad=-0.2")

    edge_labels = {(u, v): d["label"] for u, v, _, d in G.edges(keys=True, data=True) if d.get("type") == "exec"}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="darkgreen")

    legend_elements = [
        Patch(facecolor=color, edgecolor='black', label=phase)
        for phase, color in phase_colors.items()
    ]
    plt.legend(handles=legend_elements, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))

    title = f"{G.graph.get('instance_name', 'instance_id')} - {G.graph.get('resolution_status', 'unknown')}"
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    instance_dir = "trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/astropy__astropy-8872"
    eval_report_path = "sb-cli-reports/Subset.swe_bench_verified__test__evaluate_swev.json"

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

    output_prefix = traj_file.replace("trajectories", "graphs").replace(".traj", "")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    parser = CommandParser()
    parser.load_tool_yaml_files([
        "tools/edit_anthropic/config.yaml",
        "tools/review_on_submit_m/config.yaml",
        "tools/registry/config.yaml"
    ])

    build_graph_from_trajectory(traj_data, parser, output_prefix, eval_report_path)
