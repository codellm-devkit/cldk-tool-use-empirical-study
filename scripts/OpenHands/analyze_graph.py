import os
import json
import networkx as nx
from statistics import mean
from gsppy.gsp import GSP
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
        loc = self.get_localization_summary()
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
            "complex_command_freq": self.get_complex_command_frequency(),
            "loc_focus_ratio": loc["focus_ratio"],
            "loc_dominant_zone": loc["dominant_zone"],
            "loc_cluster_num": loc["num_clusters"],
            "loc_avg_node_freq": loc["loc_avg_node_freq"],
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

    def get_complex_command_frequency(self):
        return sum(1 for _, data in self.graph.nodes(data=True) if data.get("label") == "complex_command")

    def has_complex_command(self) -> bool:
        return any(data.get("label") == "complex_command" for _, data in self.graph.nodes(data=True))

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

    def get_total_execution_time(self):
        total = 0.0
        for _, data in self.graph.nodes(data=True):
            total += sum(data.get("execution_time", []))
        return round(total, 2)
    
    def get_localization_summary(self):
        """
        Returns:
        - focus_ratio: percent of actions that are localization
        - dominant_zone: early/middle/late/mixed/none
        - num_clusters: number of contiguous localization clusters
        - loc_avg_node_freq: mean revisit frequency of unique localization nodes
        """
        # Ordered phases
        steps = []
        loc_nodes_freq = []
        for node_id, data in self.graph.nodes(data=True):
            phase = data.get("phase", "general")
            freq = len(data.get("step_indices", []))
            if phase == "localization":
                loc_nodes_freq.append(freq)
            for idx in data.get("step_indices", []):
                steps.append((idx, phase))
        steps.sort(key=lambda x: x[0])
        phases = [p for _, p in steps]

        total_actions = len(phases)
        if total_actions == 0:
            return dict(focus_ratio=0, dominant_zone="none", num_clusters=0, loc_avg_node_freq=0)

        # Bins + clusters
        bins = [0, 0, 0]
        clusters, current = [], 0
        for i, p in enumerate(phases):
            if p == "localization":
                if i < total_actions // 3: bins[0] += 1
                elif i < 2 * total_actions // 3: bins[1] += 1
                else: bins[2] += 1
                current += 1
            else:
                if current > 0: clusters.append(current)
                current = 0
        if current > 0: clusters.append(current)

        total_loc = sum(bins)
        ratio = round(total_loc / total_actions, 2)

        if total_loc == 0:
            dominant = "none"
        else:
            max_b = max(bins)
            dominant = ["early", "middle", "late"][bins.index(max_b)] if max_b / total_loc >= 0.5 else "mixed"

        loc_avg_freq = round(mean(loc_nodes_freq), 2) if loc_nodes_freq else 0

        return dict(
            focus_ratio=ratio,
            dominant_zone=dominant,
            num_clusters=len(clusters),
            loc_avg_node_freq=loc_avg_freq
        )


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
    instance_dir = "../../OpenHands/graphs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/"
    out_dir = os.path.join(instance_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    difficulty_rename = {
        "<15 min fix": "easy",
        "15 min - 1 hour": "medium",
        "1-4 hours": "hard",
        ">4 hours": "very_hard"
    }

    categories = defaultdict(lambda: {
        "phases": [],
        "labels": [],
        "metrics": [],
        "freq_nodes": Counter(),
        "graph_with_complex_command_count": 0,
        "dominant_zones": [],
        "top_loc_info": []
    })
    rows = []

    for root, _, files in os.walk(instance_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(root, fname)) as f:
                data = json.load(f)

            analyzer = TrajectoryGraphAnalyzer(data)
            metrics = analyzer.get_metric_dict()
            phases = analyzer.extract_phase_sequence()
            labels = analyzer.extract_label_sequence()
            mf_node = metrics["most_freq_node"]
            freq_node = mf_node.split(":")[-1].strip() if mf_node else "Unknown"
            has_complex = analyzer.has_complex_command()

            resolution = data.get("graph", {}).get("resolution_status", "unknown")
            difficulty_raw = data.get("graph", {}).get("difficulty", "unknown")
            difficulty = difficulty_rename.get(difficulty_raw, difficulty_raw)
            patch_difficulty = data.get("graph", {}).get("patch_difficulty", "unknown")

            inst_id = data.get("graph", {}).get("instance_name")
            avg_loc_freq = metrics["loc_avg_node_freq"]
            loc_focus_ratio = metrics["loc_focus_ratio"]
            dominant = metrics["loc_dominant_zone"]

            keys = [resolution, difficulty, f"{resolution}_{difficulty}"]
            for k in keys:
                categories[k]["phases"].append(phases)
                categories[k]["labels"].append(labels)
                categories[k]["metrics"].append(metrics)
                categories[k]["freq_nodes"][freq_node] += 1
                categories[k]["dominant_zones"].append(dominant)
                if avg_loc_freq > 1:
                    categories[k]["top_loc_info"].append((inst_id, avg_loc_freq, loc_focus_ratio, dominant))
                if has_complex:
                    categories[k]["graph_with_complex_command_count"] += 1

            metrics["difficulty"] = difficulty
            metrics["patch_difficulty"] = patch_difficulty
            metrics["resolution"] = resolution
            metrics["instance"] = inst_id
            rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "trajectory_metrics.csv"), index=False)

    for k, v in categories.items():
        filename = k.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("/", "-")
        with open(os.path.join(out_dir, f"{filename}.txt"), "w") as fout:
            fout.write(f"===== Category: {k} ({len(v['phases'])} instances) =====\n")

            if not v["metrics"]:
                fout.write("# No data\n")
                continue

            df_k = pd.DataFrame(v["metrics"])

            # ---------- General Metrics ----------
            fout.write("\n-- Averaged General Metrics --\n")
            drop_cols = ["most_freq_node", "instance",
                         "loc_focus_ratio", "loc_dominant_zone",
                         "loc_cluster_num", "loc_avg_node_freq"]
            general_avg = df_k.drop(columns=drop_cols).mean(numeric_only=True).round(2)
            # for val in general_avg.values:
            #     fout.write(f"{val}\n")
            fout.write(general_avg.to_string())

            fout.write(f"\nGraphs With complex_command: {v['graph_with_complex_command_count']}\n")

            # ---------- Localization Stats ----------
            fout.write("\n\n-- Localization Focus & Clusters --\n")
            mean_focus = df_k["loc_focus_ratio"].mean().round(2)
            mean_clusters = df_k["loc_cluster_num"].mean().round(2)

            zone_counter = Counter(v["dominant_zones"])
            total = sum(zone_counter.values())
            zone_percents = {zone: f"{(count/total*100):.1f}%" for zone, count in zone_counter.items()}

            fout.write(f"Mean Localization Focus Ratio: {mean_focus}\n")
            fout.write(f"Mean Number of Localization Clusters: {mean_clusters}\n")
            fout.write("Dominant Zone Distribution:\n")
            for zone in ["early", "middle", "late", "mixed", "none"]:
                percent = zone_percents.get(zone, "0.0%")
                fout.write(f"  {zone}: {percent}\n")

            # ---------- Top Instances by Localization Frequency ----------
            fout.write("\nTop Instances with High Localization Node Frequency (>1):\n")
            top_freqs = sorted(v["top_loc_info"], key=lambda x: x[1], reverse=True)[:5]
            if not top_freqs:
                fout.write("  None found.\n")
            else:
                for inst, freq, focus, dom in top_freqs:
                    fout.write(f"  {inst} | AvgFreq: {freq:.2f} | LocPercent: {focus:.2f} | Dominant: {dom}\n")

            # ---------- Phase Patterns ----------
            fout.write("\n-- Phase Patterns --\n")
            pm = SequentialPatternMiner(v["phases"])
            ppat = pm.find_frequent_patterns()
            for p, f in pm.get_most_frequent_patterns(ppat):
                fout.write(f"Most Frequent: {p}: {f}\n")
            for p, f in pm.get_longest_patterns(ppat):
                fout.write(f"Longest: {p}: {f}\n")

            # ---------- Action Patterns ----------
            fout.write("\n-- Action Patterns --\n")
            am = SequentialPatternMiner(v["labels"])
            apat = am.find_frequent_patterns()
            for p, f in am.get_most_frequent_patterns(apat):
                fout.write(f"Most Frequent: {p}: {f}\n")
            for p, f in am.get_longest_patterns(apat):
                fout.write(f"Longest: {p}: {f}\n")

            # ---------- Most Frequent Nodes ----------
            fout.write("\n-- Most Frequent Nodes --\n")
            for label, count in v["freq_nodes"].most_common():
                fout.write(f"{label}: {count}\n")