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

# Patch difficulty lookup
golden_patch_metrics = "../golden_patch_metrics.jsonl"
if os.path.exists(golden_patch_metrics):
    with open(golden_patch_metrics, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
        golden_patch_difficulty_lookup = {item["instance_id"]: item["patch_difficulty"] for item in lines}
        golden_files_change_lookup = {item["instance_id"]: item["file_count"] for item in lines}
else:
    golden_patch_difficulty_lookup = {}
    golden_files_change_lookup = {}

def hash_node_signature(label, args):
    normalized = json.dumps({"label": label, "args": args}, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

def get_phase(tool: str, subcommand: str, command: str, label: str) -> str:
    if tool == "str_replace_editor" and subcommand == "view":
        return "localization"
    if tool == "str_replace_editor" and subcommand == "str_replace":
        return "patch"
    if command == "python" or (tool == "str_replace_editor" and subcommand == "create"):
        return "verification"
    if tool == "finish":
        return "finish"
    return "general"

def check_edit_status(tool: str, subcommand: str, content: str) -> str:
    if tool != "str_replace_editor" or subcommand != "str_replace" or not content:
        return None

    if "has been edited." in content:
        return "success"
    elif "did not appear verbatim" in content:
        return "failure: not found"
    elif "Multiple occurrences of old_str" in content:
        return "failure: multiple occurrences"
    elif "old_str" in content and "is the same as new_str" in content:
        return "failure: no change"
    else:
        return "failure: unknown"

def determine_resolution_status(instance_id, eval_report_path):
    with open(eval_report_path, 'r') as f:
        report = json.load(f)
    if instance_id in report.get("resolved_ids", []):
        return "resolved"
    elif instance_id in report.get("unresolved_ids", []):
        return "unresolved"
    return "unsubmitted"

def build_graph_from_trajectory(trajectory, parser, output_dir, eval_report_path, patch_metrics_path):
    """
    Build a graph from the trajectory data using the command parser.
    """
    G = nx.MultiDiGraph()
    node_signature_to_key = {}
    previous_node = None
    localization_nodes = []

    if os.path.exists(patch_metrics_path):
        with open(patch_metrics_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            patch_difficulty_lookup = {item["instance_id"]: item["patch_difficulty"] for item in lines}
            files_change_lookup = {item["instance_id"]: item["file_count"] for item in lines}
    else:
        patch_difficulty_lookup = {}
        files_change_lookup = {}

    step_idx = 0
    for step in trajectory.get("history", []):
        action = step.get("observation") if step.get("observation") else None
        if action in ("system", "message") or action is None:
            continue

        tool_calls = step.get("tool_call_metadata", {}).get("model_response", {}).get("choices", [])
        if not tool_calls and "tool_call_metadata" in step:
            tool_calls = [step["tool_call_metadata"]]

        parsed_commands = []
        for call in tool_calls:
            function_call = None
            if isinstance(call, dict):
                if "function" in call:
                    function_call = call["function"]
                elif "message" in call and "tool_calls" in call["message"]:
                    for tc in call["message"]["tool_calls"]:
                        if "function" in tc:
                            function_call = tc["function"]

            if not function_call:
                continue

            tool_name = function_call.get("name")
            args_raw = function_call.get("arguments", "{}")

            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}

            if action == "run":
                cmd = args.get("command", "").strip()
                parsed_commands = parser.parse(cmd)
                if not parsed_commands:
                    continue
            else:
                subcommand = args.pop("command", None)  # remove 'command' key from args
                parsed_commands = [{
                    "tool": tool_name,
                    "subcommand": subcommand,
                    "args": args
                }]
            # parsed_actions.append(parsed_commands)

        if not parsed_commands:
            continue
        
        for parsed in parsed_commands:
            tool = parsed.get("tool", "")
            if tool == "think":
                node_label = "empty action"
                node_signature = hash_node_signature(node_label, {})
                if node_signature in node_signature_to_key:
                    node_key = node_signature_to_key[node_signature]
                    G.nodes[node_key]["step_indices"].append(step_idx)
                else:
                    node_key = f"{len(G.nodes)}:{node_label}"
                    G.add_node(node_key, label=node_label, args={}, phase="general", step_indices=[step_idx])
                    node_signature_to_key[node_signature] = node_key
                if previous_node:
                    G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")
                previous_node = node_key
                continue

            subcommand = parsed.get("subcommand", "")
            command = parsed.get("command", "")
            args = parsed.get("args", {})

            if tool:
                node_label = f"{tool}\n{subcommand}" if subcommand else tool
            else:
                node_label = command or action_str.strip()

            phase = get_phase(tool, subcommand, command, node_label)
            edit_status = check_edit_status(tool, subcommand, step.get("content", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_signature = hash_node_signature(node_label, args)
            if node_signature in node_signature_to_key:
                node_key = node_signature_to_key[node_signature]
                G.nodes[node_key]["step_indices"].append(step_idx)
            else:
                node_key = f"{len(G.nodes)}:{node_label}"
                G.add_node(node_key, label=node_label, args=args, phase=phase, step_indices=[step_idx])
                node_signature_to_key[node_signature] = node_key
                if phase == "localization":
                    localization_nodes.append(node_key)

            if previous_node:
                G.add_edge(previous_node, node_key, label=str(step_idx), type="exec")
            previous_node = node_key
            
        step_idx += 1
    
    build_hierarchical_edges(G, localization_nodes)
    instance_name = trajectory.get("instance_id", "")
    output = os.path.join(output_dir, instance_name)
    os.makedirs(output, exist_ok=True)

    resolution_status = determine_resolution_status(instance_name, eval_report_path)
    G.graph["resolution_status"] = resolution_status
    G.graph["instance_name"] = instance_name
    G.graph["difficulty"] = difficulty_lookup.get(instance_name, "unknown")
    G.graph["golden_patch_difficulty"] = golden_patch_difficulty_lookup.get(instance_name, "unknown")
    G.graph["golden_files_change"] = golden_files_change_lookup.get(instance_name, 0)
    G.graph["patch_difficulty"] = patch_difficulty_lookup.get(instance_name, "unknown")
    G.graph["files_change"] = files_change_lookup.get(instance_name, 0)

    with open(os.path.join(output, f"{instance_name}.json"), "w") as f:
        json.dump(json_graph.node_link_data(G, edges="edges"), f, indent=2)
    
    _draw_graph(G, output + f"/{instance_name}.pdf")

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
            elif (
                isinstance(view_range, (list, tuple)) and
                len(view_range) == 2 and
                all(isinstance(x, int) for x in view_range)
            ):
                range_nodes_by_path[str(path_obj)].append((node, view_range))
            else:
                print(f"[WARN] Skipping invalid view_range for node {node}: {view_range}")

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
                try:
                    a1, a2 = r_i
                    b1, b2 = r_j
                    if b1 >= a1 and b2 <= a2:
                        G.add_edge(node_i, node_j, type="hier")
                        is_nested[node_j] = True
                except Exception as e:
                    print(f"[WARN] Failed to unpack ranges for nesting check: {r_i}, {r_j} ({e})")

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
        "finish": "mediumpurple",
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
    # title = f"{G.graph.get('instance_name', 'instance_id')}"
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    trajs_file = "../../OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/sample_output.jsonl"
    eval_report_path = os.path.join(
        "/".join(trajs_file.split("/")[:-1]),
        "report.json"
    )
    patch_metrics_path = os.path.join(
        "/".join(trajs_file.split("/")[:-1]),
        "patch_metrics.jsonl"
    )
    graph_dir = "/".join(trajs_file.split("/")[:-1]).replace("evaluation/evaluation_outputs/outputs/", "graphs/")
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(eval_report_path):
        print(f"Evaluation report not found at {eval_report_path}. Exiting.")
        sys.exit(1)

    parser = CommandParser()
    with open(trajs_file) as f:
        for idx, line in enumerate(f):
            traj = json.loads(line)
            if not traj:
                continue
            print(f"{idx + 1}: Processing trajectory with ID {traj.get('instance_id', 'unknown')}")
            build_graph_from_trajectory(traj, parser, graph_dir, eval_report_path, patch_metrics_path)