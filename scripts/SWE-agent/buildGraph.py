import json
import os
import sys
import hashlib
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from networkx.readwrite import json_graph
from commandParser import CommandParser
from datasets import load_dataset
from collections import defaultdict

# Load SWE-bench_Verified difficulty mapping
swe_bench_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
difficulty_lookup = {row["instance_id"]: row["difficulty"] for row in swe_bench_ds}

def hash_node_signature(label, args, state):
    normalized = json.dumps({"label": label, "args": args, "state": state}, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

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
    node_signature_to_key = {}
    trajectory = traj_data.get("trajectory", [])
    previous_node = None
    localization_nodes = []

    for step_idx, step in enumerate(trajectory):
        action_str = step.get("action", "")
        execution_time = step.get("execution_time", 0.0)
        state = step.get("state", {})

        if action_str.strip() == "":
            node_label = "empty action"
            node_signature = hash_node_signature(node_label, {}, state)
            if node_signature in node_signature_to_key:
                node_key = node_signature_to_key[node_signature]
                if previous_node:
                    G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")
                previous_node = node_key
                continue
            node_key = f"{len(G.nodes)}:{node_label}"
            G.add_node(node_key, label=node_label, execution_time=[execution_time], state=state, args={}, phase="general", step_indices=[step_idx])
            node_signature_to_key[node_signature] = node_key
            if previous_node:
                G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")
            previous_node = node_key
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

            node_signature = hash_node_signature(node_label, args, state)
            if node_signature in node_signature_to_key:
                node_key = node_signature_to_key[node_signature]
                G.nodes[node_key]["execution_time"].append(time_per_command)
                G.nodes[node_key]["step_indices"].append(step_idx)
                if previous_node:
                    G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")
                previous_node = node_key
                continue

            node_key = f"{len(G.nodes)}:{node_label}"
            G.add_node(
                node_key,
                label=node_label,
                execution_time=[time_per_command],
                state=state,
                args=args,
                phase=phase,
                step_indices=[step_idx]
            )
            node_signature_to_key[node_signature] = node_key

            if phase == "localization":
                localization_nodes.append(node_key)

            if previous_node:
                G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")
            previous_node = node_key

    build_hierarchical_edges(G, localization_nodes)

    instance_name = os.path.basename(output_prefix)
    resolution_status = determine_resolution_status(output_prefix, eval_report_path)
    G.graph["resolution_status"] = resolution_status
    G.graph["instance_name"] = instance_name
    G.graph["difficulty"] = difficulty_lookup.get(instance_name, "unknown")

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    with open(output_prefix + ".json", "w") as f:
        json.dump(json_graph.node_link_data(G, edges="edges"), f, indent=2)

    _draw_graph(G, output_prefix + ".pdf")

def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    path_nodes = []  # [(node_id, Path)]
    range_nodes_by_path = defaultdict(list)  # path_str -> [(node_id, [start, end])]

    for node in localization_nodes:
        data = G.nodes[node]
        path = data.get("args", {}).get("path")
        view_range = data.get("args", {}).get("view_range")
        if path:
            path_obj = Path(path)
            if view_range is None:
                path_nodes.append((node, path_obj))
            else:
                range_nodes_by_path[str(path_obj)].append((node, view_range))

    # --- 1) Path hierarchy by folder containment ---
    for child_node, child_path in path_nodes:
        best_parent_node = None
        best_parent_path = None
        for parent_node, parent_path in path_nodes:
            if parent_node == child_node:
                continue
            if (len(parent_path.parts) < len(child_path.parts) and
                child_path.parts[:len(parent_path.parts)] == parent_path.parts):
                if best_parent_path is None or len(parent_path.parts) > len(best_parent_path.parts):
                    best_parent_node = parent_node
                    best_parent_path = parent_path
        if best_parent_node:
            G.add_edge(best_parent_node, child_node, type="hier")

    # --- 2) Range nodes: handle nesting + link outermost ---
    path_to_node = {str(p): n for n, p in path_nodes}

    for path_str, range_nodes in range_nodes_by_path.items():
        is_nested = {n: False for n, _ in range_nodes}

        # detect nesting: mark inner ranges
        for i, (node_i, r_i) in enumerate(range_nodes):
            for j, (node_j, r_j) in enumerate(range_nodes):
                if i == j:
                    continue
                a1, a2 = r_i
                b1, b2 = r_j
                if b1 >= a1 and b2 <= a2:
                    G.add_edge(node_i, node_j, type="hier")
                    is_nested[node_j] = True

        # link outermost ranges to:
        #   - exact path node if exists
        #   - else closest parent path node whose path contains this path
        path_node = path_to_node.get(path_str)
        if path_node:
            for node, _ in range_nodes:
                if not is_nested[node]:
                    G.add_edge(path_node, node, type="hier")
        else:
            # No exact path node → find nearest ancestor
            path_parts = Path(path_str).parts
            best_ancestor_node = None
            best_ancestor_depth = -1
            for pn, pp in path_nodes:
                if len(pp.parts) < len(path_parts) and path_parts[:len(pp.parts)] == pp.parts:
                    if len(pp.parts) > best_ancestor_depth:
                        best_ancestor_node = pn
                        best_ancestor_depth = len(pp.parts)
            for node, _ in range_nodes:
                if not is_nested[node] and best_ancestor_node:
                    G.add_edge(best_ancestor_node, node, type="hier")

def _draw_graph(G: nx.MultiDiGraph, png_path: str):
    from matplotlib.patches import Patch
    plt.rcParams['pdf.fonttype'] = 42  # Embed fonts as TrueType
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(max(12, len(G.nodes) * 0.6), 10))

    try:
        from networkx.drawing.nx_agraph import graphviz_layout
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
        label = data["label"]
        args = data.get("args", {})
        status = args.get("edit_status") if isinstance(args, dict) else None
        if status == "success":
            label += " ✓"
        elif status and status.startswith("failure"):
            label += " ✗"
        labels[node] = label
        node_colors.append(color)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    for u, v, k, d in G.edges(keys=True, data=True):
        style = "solid"
        color = "gray"
        rad = 0.1 + 0.1 * (k % 3)

        if d.get("type") == "hier":
            style = "dashed"
            color = "green"
            rad = -0.2 - 0.05 * (k % 2)

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
            style=style,
            edge_color=color,
            arrows=True
        )

        if d.get("type") == "exec" and "label" in d:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            nx_offset = -(y2 - y1) * 0.03
            ny_offset = (x2 - x1) * 0.03
            plt.text(mx + nx_offset, my + ny_offset, d["label"],
                     fontsize=8, color="darkgreen", ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='none', pad=0.1))

    legend_elements = [Patch(facecolor=color, edgecolor='black', label=phase) for phase, color in phase_colors.items()]
    plt.legend(handles=legend_elements, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))

    title = f"{G.graph.get('instance_name', 'instance_id')} - {G.graph.get('resolution_status', 'unknown')}"
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    instance_dir = "../../SWE-agent/trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/astropy__astropy-7671"
    eval_report_path = "../../SWE-agent/sb-cli-reports/Subset.swe_bench_verified__test__evaluate_swev.json"

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

    output_prefix = traj_file.replace("trajectories", "graphs/samples").replace(".traj", "")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    parser = CommandParser()
    parser.load_tool_yaml_files([
        "../../SWE-agent/tools/edit_anthropic/config.yaml",
        "../../SWE-agent/tools/review_on_submit_m/config.yaml",
        "../../SWE-agent/tools/registry/config.yaml"
    ])

    build_graph_from_trajectory(traj_data, parser, output_prefix, eval_report_path)
