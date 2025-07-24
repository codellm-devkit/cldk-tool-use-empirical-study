import os
import json
import networkx as nx
from statistics import mean
from gsppy.gsp import GSP
import getpass
import pandas as pd
from collections import defaultdict, Counter
from statistics import mean

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
            # "in_degree_most_freq": self.get_in_degree(mf_node),
            # "out_degree_most_freq": self.get_out_degree(mf_node),
            "loc_focus_ratio": loc["focus_ratio"],
            "loc_dominant_zone": loc["dominant_zone"],
            "loc_cluster_num": loc["num_clusters"],
            "loc_avg_node_freq": loc["loc_avg_node_freq"],
            # "total_execution_time": self.get_total_execution_time(),
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
    
    def get_hier_graph(self):
        G_hier = nx.DiGraph()
        # Add only localization nodes
        for node, data in self.graph.nodes(data=True):
            if data.get("phase") == "localization":
                G_hier.add_node(node, **data)
        # Add hierarchical edges only if both endpoints are localization nodes
        for u, v, d in self.graph.edges(data=True):
            if d.get("type") == "hier" and u in G_hier and v in G_hier:
                G_hier.add_edge(u, v)
        return G_hier

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

    # def get_total_execution_time(self):
    #     total = 0.0
    #     for _, data in self.graph.nodes(data=True):
    #         total += sum(data.get("execution_time", []))
    #     return round(total, 2)
    
    def get_localization_summary(self):
        """
        Returns localization behavior metrics:
        - focus_ratio: percent of actions that are localization
        - dominant_zone: early/middle/late/mixed/none
        - num_clusters: number of contiguous localization clusters
        - loc_avg_node_freq: mean revisit frequency of unique localization nodes
        - repeated_view: True if any localization node is revisited
        - max_view_depth: deepest hierarchical level visited
        - avg_view_depth: average view depth
        - max_view_span: largest number of nodes viewed at same hierarchical level
        - avg_view_span: average number of nodes viewed per level
        - scroll_behavior: True if overlapping views within same file
        - num_deep_zooms_without_edit: count of leaf nodes explored without edits
        - back_and_forth_switch: True if zigzag pattern detected in view hierarchy
        """
        steps = []
        loc_nodes_freq = []
        loc_ranges_by_path = defaultdict(list)

        # -- patch nodes
        patch_paths = set()
        for _, data in self.graph.nodes(data=True):
            label = data.get("label", "")
            if label in {"str_replace_editor\nstr_replace", "str_replace_editor\ncreate", "str_replace_editor\ninsert"}:
                path = data.get("args", {}).get("path")
                if path:
                    patch_paths.add(path)

        # -- gather step and localization info
        for node_id, data in self.graph.nodes(data=True):
            phase = data.get("phase", "")
            freq = len(data.get("step_indices", []))
            if phase == "localization":
                loc_nodes_freq.append(freq)
                view_range = data.get("args", {}).get("view_range")
                path = data.get("args", {}).get("path")
                if isinstance(view_range, (list, tuple)) and len(view_range) == 2 and path:
                    loc_ranges_by_path[path].append(tuple(view_range))
            for idx in data.get("step_indices", []):
                steps.append((idx, phase, node_id))
        steps.sort(key=lambda x: x[0])
        phases = [p for _, p, _ in steps]

        total_actions = len(phases)
        if total_actions == 0:
            return {
                "focus_ratio": 0, "dominant_zone": "none", "num_clusters": 0,
                "loc_avg_node_freq": 0, "repeated_view": False,
                "max_view_depth": 0, "avg_view_depth": 0,
                "max_view_span": 0, "avg_view_span": 0,
                "scroll_behavior": False, "num_deep_zooms_without_edit": 0
            }

        # -- zone and cluster analysis
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
        dominant = (
            "none" if total_loc == 0 else
            ["early", "middle", "late"][bins.index(max(bins))] if max(bins) / total_loc >= 0.5
            else "mixed"
        )

        loc_avg_freq = round(mean(loc_nodes_freq), 2) if loc_nodes_freq else 0
        repeated_view = any(freq > 1 for freq in loc_nodes_freq)

        # --- Hierarchical structure
        hier_graph = self.get_hier_graph()
        prefix_map = {}
        node_path_map = {}

        def dfs(node, prefix):
            prefix_map[node] = prefix
            for i, child in enumerate(hier_graph.successors(node)):
                dfs(child, f"{prefix}-{i}" if prefix else str(i))

        roots = [n for n in hier_graph.nodes if hier_graph.in_degree(n) == 0]
        for i, root in enumerate(roots):
            dfs(root, str(i))

        for node_id, data in hier_graph.nodes(data=True):
            path = data.get("args", {}).get("path")
            if path:
                node_path_map[node_id] = path

        # --- View depth and span analysis ---
        exec_prefixes = [(nid, prefix_map.get(nid, "")) for _, phase, nid in steps if phase == "localization"]
        level_counts = Counter()
        leaf_depths = []

        for nid, prefix in exec_prefixes:
            if not prefix:
                continue
            level = prefix.count("-")
            level_counts[level] += 1

            # Check if it's a leaf in the hierarchy
            if hier_graph.out_degree(nid) == 0:
                leaf_depths.append(level + 1)  # depth = number of segments

        max_view_span = max(level_counts.values(), default=0)
        avg_view_span = round(mean(level_counts.values()), 2) if level_counts else 0
        max_view_depth = max(leaf_depths, default=0)
        avg_view_depth = round(mean(leaf_depths), 2) if leaf_depths else 0

        # --- scroll behavior
        scroll_behavior = False
        for path, ranges in loc_ranges_by_path.items():
            if len(ranges) <= 1:
                continue
            ranges.sort()
            for i in range(1, len(ranges)):
                if ranges[i][0] <= ranges[i - 1][1]:
                    scroll_behavior = True
                    break
            if scroll_behavior:
                break

        # --- deep zoom without edit
        leaf_nodes = [n for n in hier_graph.nodes if hier_graph.out_degree(n) == 0]
        leaf_paths = {
            node_path_map[n] for n in leaf_nodes
            if n in node_path_map and n in prefix_map
        }
        deep_zooms_without_edit = [
            p for p in leaf_paths if not any(p in patch for patch in patch_paths)
        ]
        num_deep_zooms_without_edit = len(deep_zooms_without_edit)

        # --- back-and-forth switch detection ---
        def common_prefix_len(p1, p2):
            s1, s2 = p1[1].split("-"), p2[1].split("-")
            i = 0
            while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
                i += 1
            return i

        back_and_forth_switch = False
        window = []
        for p in exec_prefixes:
            if not p:
                continue
            window.append(p)
            if len(window) > 3:
                window.pop(0)
            if len(window) == 3:
                pre, mid, post = window
                cp1 = common_prefix_len(pre, mid)
                cp2 = common_prefix_len(mid, post)
                cp3 = common_prefix_len(pre, post)
                # Detect zigzag: mid diverges from pre and post, while pre and post are similar
                if cp3 >= max(cp1, cp2) and cp1 != cp2:
                    back_and_forth_switch = True
                    break

        return {
            "focus_ratio": ratio,
            "dominant_zone": dominant,
            "num_clusters": len(clusters),
            "loc_avg_node_freq": loc_avg_freq,
            "repeated_view": repeated_view,
            "max_view_depth": max_view_depth,
            "avg_view_depth": avg_view_depth,
            "max_view_span": max_view_span,
            "avg_view_span": avg_view_span,
            "scroll_behavior": scroll_behavior,
            "num_deep_zooms_without_edit": num_deep_zooms_without_edit,
            "back_and_forth_switch": back_and_forth_switch
        }

    def get_patch_summary(self):
        """
        Analyze patching behavior from a trajectory graph.

        Returns:
            dict: A summary of patch-related metrics, including:
                - patch_total: total number of patch attempts.
                - patch_success: count of successful patch attempts.
                - fail_types: breakdown of all failure types encountered.
                - fail_streaks: dict with max, average, and count of consecutive failed patch attempts.
                - flip_flop: True if an edit is undone by a reverse change.
                - repeat_failed_edit: True if a previously failed patch is attempted and failed again.
                - abandonment: True if there is no successful patch in the entire sequence.
                - fail_to_success_patterns: common reasoning phase transitions from a failed to successful patch.
        """
        patch_nodes = []
        step_node_map = {}

        # First, extract all patch nodes and build a step-to-node map
        for node_id, data in self.graph.nodes(data=True):
            phase = data.get("phase", "")
            if phase == "patch":
                for step in data.get("step_indices", []):
                    patch_nodes.append((step, node_id, data))
            for step in data.get("step_indices", []):
                step_node_map[step] = (node_id, data)

        # Sort the patch_nodes based on step indices
        patch_nodes.sort(key=lambda x: x[0])
        patch_steps = sorted(step_node_map.keys())

        patch_total = len(patch_nodes)
        patch_success = 0
        fail_types = Counter()
        fail_streaks = []
        seen_edits = set()
        flip_flop = False
        repeat_failed_edit = False
        abandonment = False
        edit_history = []
        current_streak = 0
        reasoning_between_patches = []
        fail_to_success_phases = []
        reasoning_transitions = Counter()
        fail_success_transitions = Counter()

        previous_status = None
        previous_edit = None
        previous_step = None

        # Reasoning span detection between patches
        for i in range(len(patch_steps) - 1):
            span = list(range(patch_steps[i] + 1, patch_steps[i + 1]))
            phases = []
            for s in span:
                _, node_data = step_node_map.get(s, (None, {}))
                if node_data:
                    phase = node_data.get("phase", "unknown")
                    if phase != "patch":
                        phases.append(phase)
            if phases:
                reasoning_between_patches.append(phases)
                deduped = [p for i, p in enumerate(phases) if i == 0 or p != phases[i-1]]
                reasoning_transitions[tuple(deduped)] += 1

        for step, node_id, data in patch_nodes:
            args = data.get("args", {})
            path = args.get("path", "")
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")
            status_raw = args.get("edit_status", "")
            edit_key = (path, old_str, new_str)

            # Normalize status
            if status_raw.strip() == "success":
                status = "success"
                patch_success += 1
                if current_streak > 0:
                    fail_streaks.append(current_streak)
                    current_streak = 0
            else:
                status = status_raw.replace("failure: ", "").strip()
                current_streak += 1
                fail_types[status] += 1
                if edit_key in seen_edits:
                    repeat_failed_edit = True

            if edit_history and edit_key == (edit_history[-1][0], edit_history[-1][2], edit_history[-1][1]):
                flip_flop = True

            if previous_status != "success" and status == "success":
                if previous_step is not None:
                    span = list(range(previous_step + 1, step))
                    inter_phases = []
                    for s in span:
                        _, node_data = step_node_map.get(s, (None, {}))
                        if node_data:
                            phase = node_data.get("phase", "unknown")
                            if phase != "patch":
                                inter_phases.append(phase)
                    if inter_phases:
                        fail_to_success_phases.append(inter_phases)
                        deduped = [p for i, p in enumerate(inter_phases) if i == 0 or p != inter_phases[i-1]]
                        fail_success_transitions[tuple(deduped)] += 1

            seen_edits.add(edit_key)
            edit_history.append(edit_key)
            previous_edit = edit_key
            previous_status = status
            previous_step = step

        if current_streak > 0:
            fail_streaks.append(current_streak)
        if patch_total > 0 and patch_success == 0:
            abandonment = True

        max_fail_streak = max(fail_streaks) if fail_streaks else 0
        avg_fail_streak = round(mean(fail_streaks), 2) if fail_streaks else 0
        num_fail_streaks = len(fail_streaks)

        full_fail_types = {
            "not found": 0,
            "no change": 0,
            "multiple occurrences": 0,
            "unknown": 0
        }
        full_fail_types.update(fail_types)

        return {
            "patch_total": patch_total,
            "patch_success": patch_success,
            "fail_types": dict(full_fail_types),
            "fail_streaks": {
                "max": max_fail_streak,
                "avg": avg_fail_streak,
                "count": num_fail_streaks
            },
            "flip_flop": flip_flop,
            "repeat_failed_edit": repeat_failed_edit,
            "abandonment": abandonment,
            "fail_to_success_patterns": fail_success_transitions.most_common(3),
        }

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

# --------------------------- Localization Stats Group Summary ---------------------------
def summarize_localization_stats(localization_stats_list):
    counter_fields = [
        "repeated_view", "scroll_behavior", "back_and_forth_switch"
    ]
    numeric_fields = [
        "focus_ratio", "num_clusters", "loc_avg_node_freq",
        "max_view_depth", "avg_view_depth", "max_view_span", "avg_view_span",
        "num_deep_zooms_without_edit"
    ]

    result = {f: 0 for f in counter_fields}
    sums = {f: 0.0 for f in numeric_fields}
    n = len(localization_stats_list)

    deep_zooms_without_edit_count = 0

    for stat in localization_stats_list:
        for f in counter_fields:
            result[f] += int(stat.get(f, False))
        for f in numeric_fields:
            value = stat.get(f, 0)
            sums[f] += value
            if f == "num_deep_zooms_without_edit" and value > 1:
                deep_zooms_without_edit_count += 1

    for f in numeric_fields:
        result[f] = round(sums[f] / n, 2) if n else 0

    result["deep_zooms_without_edit"] = deep_zooms_without_edit_count

    return result

# --------------------------- Patch Stats Group Summary ---------------------------
def summarize_patch_stats(patch_stats_list):
    numeric_fields = ["patch_total", "patch_success"]
    summed_flags = ["flip_flop", "repeat_failed_edit", "abandonment"]
    streak_maxes, streak_avgs, streak_counts = [], [], []

    # reasoning_counter = Counter()
    fail_success_counter = Counter()
    fail_types_sum = Counter()

    result = {f: 0 for f in numeric_fields + summed_flags}

    for entry in patch_stats_list:
        for f in numeric_fields:
            result[f] += entry.get(f, 0)
        for f in summed_flags:
            result[f] += int(entry.get(f, False))

        fs = entry.get("fail_streaks", {})
        streak_maxes.append(fs.get("max", 0))
        streak_avgs.append(fs.get("avg", 0))
        streak_counts.append(fs.get("count", 0))

        fail_types_sum.update(entry.get("fail_types", {}))

        if entry.get("fail_to_success_patterns"):
            fail_success_counter.update([tuple(entry["fail_to_success_patterns"][0][0])])

    n = len(patch_stats_list)
    result["avg_patch_total"] = round(result["patch_total"] / n, 2) if n else 0
    result["patch_success_rate"] = round(result["patch_success"] / result["patch_total"], 2) if result["patch_total"] > 0 else 0
    result["avg_max_fail_streak"] = round(sum(streak_maxes) / n, 2) if n else 0
    result["avg_avg_fail_streak"] = round(sum(streak_avgs) / n, 2) if n else 0
    result["avg_fail_streak_count"] = round(sum(streak_counts) / n, 2) if n else 0
    result["top_fail_to_success"] = fail_success_counter.most_common(1)[0][0] if fail_success_counter else "N/A"
    result["fail_types_sum"] = dict(fail_types_sum)
    return result

def write_summary_file(filename, category_name, category_data):
    with open(filename, "w") as fout:
        fout.write(f"===== Category: {category_name} ({len(category_data['phases'])} instances) =====\n")
        if not category_data["metrics"]:
            fout.write("No data.\n")
            return

        df_k = pd.DataFrame(category_data["metrics"])

        # ---------- General Metrics ----------
        fout.write("\n-- Averaged General Metrics --\n")
        drop_cols = [
            "most_freq_node", "instance", "difficulty", "patch_difficulty", "resolution",
            "loc_focus_ratio", "loc_dominant_zone", "loc_cluster_num", "loc_avg_node_freq",
            "patch_total", "patch_success",
            "fail_streak_max", "fail_streak_avg", "fail_streak_count",
            "reasoning_transition_patterns", "fail_to_success_patterns",
            "flip_flop", "repeat_failed_edit", "abandonment",
            "repeated_view", "scroll_behavior", "back_and_forth_switch", "num_deep_zooms_without_edit",
            "avg_view_depth", "max_view_depth", "avg_view_span", "max_view_span",
        ]
        drop_cols += [col for col in df_k.columns if col.startswith("fail_type_")]
        general_avg = df_k.drop(columns=drop_cols, errors='ignore').mean(numeric_only=True).round(2)
        fout.write(general_avg.to_string())

        # ---------- Localization Focus & Clusters ----------
        fout.write("\n\n-- Localization Focus & Clusters --\n")
        mean_focus = df_k["loc_focus_ratio"].mean().round(2)
        mean_clusters = df_k["loc_cluster_num"].mean().round(2)
        zone_counter = Counter(category_data["dominant_zones"])
        total = sum(zone_counter.values())
        zone_percents = {zone: f"{(count / total * 100):.1f}%" for zone, count in zone_counter.items()}
        fout.write(f"Mean Localization Focus Ratio: {mean_focus}\n")
        fout.write(f"Mean Number of Localization Clusters: {mean_clusters}\n")
        fout.write("Dominant Zone Distribution:\n")
        for zone in ["early", "middle", "late", "mixed", "none"]:
            percent = zone_percents.get(zone, "0.0%")
            fout.write(f"  {zone}: {percent}\n")
        fout.write("\nTop Instances with High Localization Node Frequency (>1):\n")
        top_freqs = sorted(category_data["top_loc_info"], key=lambda x: x[1], reverse=True)[:5]
        if not top_freqs:
            fout.write("  None found.\n")
        else:
            for inst, freq, focus, dom in top_freqs:
                fout.write(f"  {inst} | AvgFreq: {freq:.2f} | LocPercent: {focus:.2f} | Dominant: {dom}\n")
        
        # ---------- Localization Behavior Summary ----------
        fout.write("\n-- Localization Behavior Summary --\n")
        loc_summary = summarize_localization_stats(category_data["localization"])
        for field in [
            "repeated_view", "scroll_behavior", "back_and_forth_switch",
        ]:
            percent_true = (loc_summary[field] / len(category_data["localization"])) * 100 if category_data["localization"] else 0
            fout.write(f"{field}: {loc_summary[field]}({percent_true:.1f}%) \n")
        
        deep_zooms_percent_true = (loc_summary['deep_zooms_without_edit'] / len(category_data["localization"])) * 100 if category_data["localization"] else 0
        fout.write(f"deep_zooms_without_edit: {loc_summary['deep_zooms_without_edit']}({deep_zooms_percent_true:.1f}%)\n")

        for field in [
            "max_view_depth", "avg_view_depth", "max_view_span", "avg_view_span",
            "num_deep_zooms_without_edit"
        ]:
            fout.write(f"{field}: {loc_summary[field]:.2f}\n")

        # ---------- Phase & Action Patterns ----------
        fout.write("\n-- Phase Patterns --\n")
        pm = SequentialPatternMiner(category_data["phases"])
        for p, f in pm.get_most_frequent_patterns(pm.find_frequent_patterns()):
            fout.write(f"Most Frequent: {p}: {f}\n")
        for p, f in pm.get_longest_patterns(pm.find_frequent_patterns()):
            fout.write(f"Longest: {p}: {f}\n")
        fout.write("\n-- Action Patterns --\n")
        am = SequentialPatternMiner(category_data["labels"])
        for p, f in am.get_most_frequent_patterns(am.find_frequent_patterns()):
            fout.write(f"Most Frequent: {p}: {f}\n")
        for p, f in am.get_longest_patterns(am.find_frequent_patterns()):
            fout.write(f"Longest: {p}: {f}\n")

        fout.write("\n-- Most Frequent Nodes --\n")
        for label, count in category_data["freq_nodes"].most_common():
            fout.write(f"{label}: {count}\n")

        # ---------- Patch Behavior Metrics ----------
        fout.write("\n-- Patch Behavior Summary --\n")
        patch_summary = summarize_patch_stats(category_data["patches"])
        fout.write(f"Avg Patch Total: {patch_summary['avg_patch_total']}\n")
        fout.write(f"Avg Patch Success Rate: {patch_summary['patch_success_rate']}\n")
        fout.write(f"Avg Max Fail Streak: {patch_summary['avg_max_fail_streak']}\n")
        fout.write(f"Avg Avg Fail Streak: {patch_summary['avg_avg_fail_streak']}\n")
        fout.write(f"Avg Fail Streak Count: {patch_summary['avg_fail_streak_count']}\n")
        fout.write(f"Total Flip-Flop Edits: {patch_summary['flip_flop']}\n")
        fout.write(f"Total Repeat Failed Edits: {patch_summary['repeat_failed_edit']}\n")
        fout.write(f"Total Abandonment Cases: {patch_summary['abandonment']}\n")
        fout.write(f"Top Fail-to-Success Transition Pattern: {patch_summary['top_fail_to_success']}\n")
        fail_order = ["not found", "no change", "multiple occurrences", "unknown"]
        for kf in fail_order:
            vf = patch_summary["fail_types_sum"].get(kf, 0)
            fout.write(f"  {kf}: {vf}\n")

# --------------------------- Main Analysis ---------------------------
if __name__ == "__main__":
    user = getpass.getuser()
    instance_dir = f"../../SWE-agent/graphs/{user}"
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
        "localization": [],
        "patches": [],
        "freq_nodes": Counter(),
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
            localization_stats = analyzer.get_localization_summary()
            patch_stats = analyzer.get_patch_summary()
            phases = analyzer.extract_phase_sequence()
            labels = analyzer.extract_label_sequence()

            mf_node = metrics["most_freq_node"]
            freq_node = mf_node.split(":")[-1].strip() if mf_node else "Unknown"

            resolution = data.get("graph", {}).get("resolution_status", "unknown")
            difficulty_raw = data.get("graph", {}).get("difficulty", "unknown")
            difficulty = difficulty_rename.get(difficulty_raw, difficulty_raw)
            patch_difficulty = data.get("graph", {}).get("patch_difficulty", "unknown")
            files_changed = "single" if data.get("graph", {}).get("files_change", 0) == 1 else "multiple"
            inst_id = data.get("graph", {}).get("instance_name")
            avg_loc_freq = metrics["loc_avg_node_freq"]
            loc_focus_ratio = metrics["loc_focus_ratio"]
            dominant = metrics["loc_dominant_zone"]

            keys = [resolution, difficulty, files_changed, f"{resolution}_{difficulty}", f"{resolution}_{files_changed}"]
            for k in keys:
                categories[k]["phases"].append(phases)
                categories[k]["labels"].append(labels)
                categories[k]["metrics"].append(metrics)
                categories[k]["localization"].append(localization_stats)
                categories[k]["patches"].append(patch_stats)
                categories[k]["freq_nodes"][freq_node] += 1
                categories[k]["dominant_zones"].append(dominant)
                if avg_loc_freq > 1:
                    categories[k]["top_loc_info"].append((inst_id, avg_loc_freq, loc_focus_ratio, dominant))

            # Flatten patch stats for CSV
            flat_patch_stats = patch_stats.copy()
            fs = flat_patch_stats.pop("fail_streaks", {})
            flat_patch_stats["fail_streak_max"] = fs.get("max", 0)
            flat_patch_stats["fail_streak_avg"] = fs.get("avg", 0)
            flat_patch_stats["fail_streak_count"] = fs.get("count", 0)

            fail_types = flat_patch_stats.pop("fail_types", {})
            for kf, vf in fail_types.items():
                flat_patch_stats[f"fail_type_{kf}"] = vf

            for k in ["fail_to_success_patterns"]:
                val = flat_patch_stats.get(k, [])
                flat_patch_stats[k] = str(val[0][0]) if val else "N/A"

            metrics.update({
                "difficulty": difficulty,
                "patch_difficulty": patch_difficulty,
                "resolution": resolution,
                "instance": inst_id
            })
            for drop_field in [
                "focus_ratio", "dominant_zone", "num_clusters", "loc_avg_node_freq"
            ]:
                localization_stats.pop(drop_field, None)
            metrics.update(localization_stats)
            metrics.update(flat_patch_stats)
            rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "trajectory_metrics.csv"), index=False)

    # Per-category summaries
    for k, v in categories.items():
        filename = k.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("/", "-")
        write_summary_file(os.path.join(out_dir, f"{filename}.txt"), k, v)

    # Overall summary across all instances
    resolutions = set(k for k in categories.keys() if "_" not in k and k in ["single", "multiple"])
    aggregated = {
        "phases": [],
        "labels": [],
        "metrics": [],
        "patches": [],
        "localization": [],
        "freq_nodes": Counter(),
        "dominant_zones": [],
        "top_loc_info": []
    }
    for r in resolutions:
        for k in categories:
            if k == r:
                for field in aggregated:
                    if isinstance(aggregated[field], list):
                        aggregated[field].extend(categories[k][field])
                    else:
                        aggregated[field] += categories[k][field]

    write_summary_file(os.path.join(out_dir, "summary_all_instances.txt"), "All Instances", aggregated)