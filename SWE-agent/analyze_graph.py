import os
import json
import networkx as nx
from statistics import mean
from gsppy.gsp import GSP
import getpass
import pandas as pd
from collections import defaultdict, Counter

# --------------------------- Graph Analyzer ---------------------------
class TrajectoryGraphAnalyzer:
    def __init__(self, graph_data):
        self.raw_data = graph_data
        self.graph = self._load_graph()

    def _load_graph(self):
        return nx.node_link_graph(self.raw_data, edges="edges")

    def get_metric_dict(self):
        mf_node = self.get_most_frequent_node()
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "exec_edge_count": len(self.get_exec_edges()),
            "loop_count": self.get_loop_count(),
            "max_loop_length": self.get_max_loop_length(),
            "min_loop_length": self.get_min_loop_length(),
            "avg_loop_length": self.get_avg_loop_length(),
            "avg_degree": self.get_avg_degree(),
            "longest_path": self.get_longest_simple_path(),
            "most_freq_node": mf_node.replace("\n", " ") if mf_node else None,
            "most_freq_node_freq": self.get_frequency(mf_node),
            "in_degree_most_freq": self.get_in_degree(mf_node),
            "out_degree_most_freq": self.get_out_degree(mf_node),
        }

    def get_exec_edges(self):
        return [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "exec"]

    def get_exec_graph(self):
        G_exec = nx.DiGraph()
        for node, data in self.graph.nodes(data=True):
            G_exec.add_node(node, **data)
        for u, v, d in self.graph.edges(data=True):
            if d.get("type") == "exec":
                G_exec.add_edge(u, v)
        return G_exec

    def get_loop_count(self):
        return sum(1 for _ in nx.simple_cycles(self.get_exec_graph()))

    def get_loop_lengths(self):
        return [len(cycle) for cycle in nx.simple_cycles(self.get_exec_graph())]

    def get_max_loop_length(self):
        lengths = self.get_loop_lengths()
        return max(lengths) if lengths else 0

    def get_min_loop_length(self):
        lengths = self.get_loop_lengths()
        return min(lengths) if lengths else 0

    def get_avg_loop_length(self):
        lengths = self.get_loop_lengths()
        return mean(lengths) if lengths else 0

    def get_avg_degree(self):
        degrees = [deg for _, deg in self.graph.degree()]
        return mean(degrees) if degrees else 0

    def get_frequency(self, node):
        return len(self.graph.nodes[node].get("step_indices", [])) if node else 0

    def get_in_degree(self, node):
        return self.graph.in_degree(node) if node else 0

    def get_out_degree(self, node):
        return self.graph.out_degree(node) if node else 0

    def get_most_frequent_node(self):
        return max(
            self.graph.nodes,
            key=lambda n: len(self.graph.nodes[n].get("step_indices", [])),
            default=None
        )

    def get_longest_simple_path(self):
        exec_graph = self.get_exec_graph()
        if nx.is_directed_acyclic_graph(exec_graph):
            path = nx.dag_longest_path(exec_graph)
            return len(path) - 1
        else:
            condensed = nx.condensation(exec_graph)
            path = nx.dag_longest_path(condensed)
            return len(path) - 1

    def extract_phase_sequence(self):
        step_sequence = []
        for node in self.graph.nodes(data=True):
            for idx in node[1].get("step_indices", []):
                step_sequence.append((idx, node[1]))
        step_sequence.sort(key=lambda x: x[0])
        seq, prev = [], None
        for _, node in step_sequence:
            curr = node.get("phase")
            if curr and curr != "general" and curr != prev:
                seq.append(curr)
                prev = curr
        return seq

    def extract_label_sequence(self):
        step_sequence = []
        for node in self.graph.nodes(data=True):
            for idx in node[1].get("step_indices", []):
                step_sequence.append((idx, node[1]))
        step_sequence.sort(key=lambda x: x[0])
        return [n.get("label", "Unknown").replace('\n', ': ').strip() for _, n in step_sequence]

# --------------------------- GSP Miner ---------------------------
class SequentialPatternMiner:
    def __init__(self, sequences):
        self.sequences = sequences

    def find_frequent_patterns(self, min_support=0.3):
        if len(self.sequences) <= 1:
            print("\u26a0\ufe0f Only one transaction. Returning self.")
            if self.sequences:
                return [{tuple(self.sequences[0]): 1}]
            else:
                return []
        gsp = GSP(self.sequences)
        return gsp.search(min_support)

    def flatten_patterns(self, patterns):
        flat = []
        for d in patterns:
            flat.extend(d.items())
        return flat

    def get_most_frequent_patterns(self, patterns):
        flat = self.flatten_patterns(patterns)
        if not flat:
            return []
        max_freq = max(freq for _, freq in flat)
        return [(pat, freq) for pat, freq in flat if freq == max_freq]

    def get_longest_patterns(self, patterns):
        flat = self.flatten_patterns(patterns)
        if not flat:
            return []
        max_len = max(len(pat) for pat, _ in flat)
        return [(pat, freq) for pat, freq in flat if len(pat) == max_len]


# --------------------------- Main Analysis ---------------------------
if __name__ == "__main__":
    user = getpass.getuser()
    instance_dir = f"graphs/{user}"
    out_dir = os.path.join(instance_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    resolve_states = ["resolved", "unresolved", "unsubmitted"]
    difficulty_rename = {
        "<15 min fix": "easy",
        "15 min - 1 hour": "medium",
        "1-4 hours": "hard",
        ">4 hours": "very_hard"
    }

    categories = defaultdict(lambda: {"phases": [], "labels": [], "metrics": [], "freq_nodes": Counter()})
    rows = []

    for root, _, files in os.walk(instance_dir):
        for fname in files:
            if not fname.endswith(".json"): continue
            with open(os.path.join(root, fname)) as f:
                data = json.load(f)
            analyzer = TrajectoryGraphAnalyzer(data)
            metrics = analyzer.get_metric_dict()
            phases = analyzer.extract_phase_sequence()
            labels = analyzer.extract_label_sequence()
            mf_node = metrics["most_freq_node"]
            freq_node = mf_node.split(":")[-1].strip() if mf_node else "Unknown"

            resolution = data.get("graph", {}).get("resolution_status", "unknown")
            difficulty_raw = data.get("graph", {}).get("difficulty", "unknown")
            difficulty = difficulty_rename.get(difficulty_raw, difficulty_raw)

            keys = [resolution, difficulty, f"{resolution}_{difficulty}"]
            for k in keys:
                categories[k]["phases"].append(phases)
                categories[k]["labels"].append(labels)
                categories[k]["metrics"].append(metrics)
                categories[k]["freq_nodes"][freq_node] += 1

            metrics["resolution"] = resolution
            metrics["difficulty"] = difficulty
            metrics["instance"] = data.get("graph", {}).get("instance_name")
            rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "trajectory_metrics.csv"), index=False)

    for k, v in categories.items():
        filename = k.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("/", "-")
        with open(os.path.join(out_dir, f"{filename}.txt"), "w") as fout:
            fout.write(f"===== Category: {k} ({len(v['phases'])} instances) =====\n")
            if not v["metrics"]:
                fout.write("No data.\n")
                continue

            fout.write("\n-- Averaged Metrics --\n")
            df_k = pd.DataFrame(v["metrics"])
            fout.write(df_k.drop(columns=["most_freq_node", "instance"]).mean(numeric_only=True).to_string())

            fout.write("\n\n-- Phase Patterns --\n")
            pm = SequentialPatternMiner(v["phases"])
            ppat = pm.find_frequent_patterns()
            for p, f in pm.get_most_frequent_patterns(ppat):
                fout.write(f"Most Frequent: {p}: {f}\n")
            for p, f in pm.get_longest_patterns(ppat):
                fout.write(f"Longest: {p}: {f}\n")

            fout.write("\n-- Action Patterns --\n")
            am = SequentialPatternMiner(v["labels"])
            apat = am.find_frequent_patterns()
            for p, f in am.get_most_frequent_patterns(apat):
                fout.write(f"Most Frequent: {p}: {f}\n")
            for p, f in am.get_longest_patterns(apat):
                fout.write(f"Longest: {p}: {f}\n")

            fout.write("\n-- Most Frequent Nodes --\n")
            for label, count in v["freq_nodes"].most_common():
                fout.write(f"{label}: {count}\n")