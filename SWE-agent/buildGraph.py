import json
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from commandParser import CommandParser
from pathlib import Path


def build_graph_from_trajectory(traj_data, parser: CommandParser, output_prefix: str):
    G = nx.MultiDiGraph()
    trajectory = traj_data.get("trajectory", [])
    previous_node = None
    node_counter = 0
    localization_nodes = []

    for step_idx, step in enumerate(trajectory):
        action_str = step.get("action", "")
        execution_time = step.get("execution_time", 0.0)
        state = step.get("state", {})

        # Handle empty action
        if action_str.strip() == "":
            node_label = "empty action"
            node_key = f"{node_counter}:{node_label}"
            G.add_node(
                node_key,
                label=node_label,
                execution_time=execution_time,
                state=state,
                args={},
                color="lightgray"
            )
            if previous_node is not None:
                G.add_edge(
                    previous_node, node_key,
                    label=str(step_idx),
                    type="exec"
                )
            previous_node = node_key
            node_counter += 1
            continue

        parsed_commands = parser.parse(action_str)
        if not parsed_commands:
            continue

        num_subcommands = len(parsed_commands)
        time_per_command = execution_time / num_subcommands if num_subcommands > 0 else 0

        for i, parsed in enumerate(parsed_commands):
            if parsed is None:
                node_label = action_str.strip()
                args = {}
            else:
                if "tool" in parsed:
                    tool = parsed["tool"]
                    subcommand = parsed.get("subcommand", "")
                    node_label = f"{tool}\n{subcommand}" if subcommand else tool
                else:
                    node_label = parsed["command"]
                args = parsed.get("args", {})

            is_localization = node_label.startswith("str_replace_editor\nview")
            node_color = "lightcoral" if is_localization else "skyblue"

            node_key = f"{node_counter}:{node_label}"
            G.add_node(
                node_key,
                label=node_label,
                execution_time=time_per_command,
                state=state,
                args=args,
                color=node_color
            )

            if is_localization:
                G.nodes[node_key]["path"] = args.get("path", "")
                localization_nodes.append(node_key)

            if previous_node is not None:
                G.add_edge(
                    previous_node, node_key,
                    label=str(step_idx),
                    type="exec"
                )

            previous_node = node_key
            node_counter += 1

    build_hierarchical_edges(G, localization_nodes)

    json_path = output_prefix + ".json"
    with open(json_path, "w") as f:
        json.dump(json_graph.node_link_data(G, edges="edges"), f, indent=2)

    _draw_graph(G, output_prefix + ".png")
    print(f"Graph saved to {json_path} and {output_prefix}.png")


def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    view_nodes = []

    for node in localization_nodes:
        data = G.nodes[node]
        path = data.get("path")
        if path:
            view_nodes.append((node, Path(path), "view_range" in data.get("args", {})))

    view_nodes.sort(key=lambda x: len(x[1].parts))

    path_to_node = {}
    for node, path, is_range in view_nodes:
        if not is_range:
            path_to_node[str(path)] = node

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
    plt.figure(figsize=(max(12, len(G.nodes) * 0.8), 10))

    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G, seed=42)

    node_colors = [d.get("color", "skyblue") for _, d in G.nodes(data=True)]
    labels = {n: d["label"] for n, d in G.nodes(data=True)}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    exec_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("type") == "exec"]
    hier_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("type") == "hier"]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in exec_edges],
        edge_color="gray",
        arrows=True,
        connectionstyle="arc3,rad=0.0"
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in hier_edges],
        edge_color="green",
        style="dashed",
        arrows=True,
        connectionstyle="arc3,rad=-0.3"
    )

    edge_labels = {
        (u, v): d["label"]
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("type") == "exec" and "label" in d
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="darkgreen")

    plt.title("Trajectory Graph", fontsize=14)
    plt.axis("off")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    instance_dir = "trajectories/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/astropy__astropy-13398"
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

    build_graph_from_trajectory(traj_data, parser, output_prefix)