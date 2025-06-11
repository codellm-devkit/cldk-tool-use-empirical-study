import os
import json
import networkx as nx
from statistics import mean
from gsppy.gsp import GSP


class TrajectoryGraphAnalyzer:
    def __init__(self, graph_data):
        self.raw_data = graph_data
        self.graph = self._load_graph()

    def _load_graph(self):
        return nx.node_link_graph(self.raw_data, edges="edges")

    def get_node_count(self):
        return self.graph.number_of_nodes()

    def get_edge_count(self):
        return self.graph.number_of_edges()

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
        return len(self.graph.nodes[node].get("step_indices", []))

    def get_in_degree(self, node):
        return self.graph.in_degree(node)

    def get_out_degree(self, node):
        return self.graph.out_degree(node)

    def get_most_frequent_node(self):
        return max(
            self.graph.nodes,
            key=lambda n: len(self.graph.nodes[n].get("step_indices", [])),
            default=None
        )

    def get_longest_path(self):
        exec_graph = self.get_exec_graph()
        if nx.is_directed_acyclic_graph(exec_graph):
            path = nx.dag_longest_path(exec_graph)
            return len(path)
        else:
            condensed = nx.condensation(exec_graph)
            path = nx.dag_longest_path(condensed)
            return len(path)

    def aggregate_nodes_by_step_index(self):
        step_sequence = []
        for node in self.graph.nodes(data=True):
            step_indices = node[1].get("step_indices", [])
            for idx in step_indices:
                step_sequence.append((idx, node[1]))
        step_sequence.sort(key=lambda x: x[0])
        ordered_nodes = [node for _, node in step_sequence]
        return ordered_nodes

    def extract_phase_sequence(self):
        ordered_nodes = self.aggregate_nodes_by_step_index()
        phase_sequence = []
        prev_phase = None
        for node in ordered_nodes:
            curr_phase = node.get('phase')
            if curr_phase and curr_phase != "general" and curr_phase != prev_phase:
                phase_sequence.append(curr_phase)
                prev_phase = curr_phase
        return phase_sequence

    def extract_label_sequence(self):
        ordered_nodes = self.aggregate_nodes_by_step_index()
        label_sequence = []
        for node in ordered_nodes:
            label = node.get('label', 'Unknown Node').replace('\n', ': ').strip()
            label_sequence.append(label)
        return label_sequence


class SequentialPatternMiner:
    def __init__(self, sequences):
        self.sequences = sequences

    def find_frequent_patterns(self, min_support=0.3):
        if len(self.sequences) <= 1:
            print("⚠️ Warning: Only one transaction found. GSP requires multiple transactions.")
            if self.sequences:
                return [{tuple(self.sequences[0]): 1}]
            else:
                return []
        try:
            gsp = GSP(self.sequences)
            return gsp.search(min_support)
        except ValueError as e:
            print(f"⚠️ GSP Error: {e}")
            if self.sequences:
                return [{tuple(self.sequences[0]): 1}]
            return []

    def flatten_patterns(self, patterns):
        flat = []
        for d in patterns:
            flat.extend(d.items())  # Each dict: {pattern: freq}
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


if __name__ == "__main__":
    instance_dir = "graphs/unsubmitted/shuyang/anthropic_filemap__deepseek/deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/"

    phase_sequences = []
    label_sequences = []

    for root, _, files in os.walk(instance_dir):
        for fname in files:
            if fname.endswith(".json"):
                json_path = os.path.join(root, fname)
                with open(json_path, "r") as f:
                    data = json.load(f)

                print(f"\nProcessing: {fname}")
                analyzer = TrajectoryGraphAnalyzer(data)

                phase_seq = analyzer.extract_phase_sequence()
                label_seq = analyzer.extract_label_sequence()

                print("Phase Sequence:", phase_seq)

                phase_sequences.append(phase_seq)
                label_sequences.append(label_seq)

                print("Graph Metrics:")
                print("Node Count:", analyzer.get_node_count())
                print("Edge Count:", analyzer.get_edge_count())
                print("Execution Edge Count:", len(analyzer.get_exec_edges()))        
                print("Loop Count:", analyzer.get_loop_count())
                print("Max Loop Length:", analyzer.get_max_loop_length())
                print("Min Loop Length:", analyzer.get_min_loop_length())
                print("Avg Loop Length:", analyzer.get_avg_loop_length())
                print("Avg Degree:", analyzer.get_avg_degree())
                print("Most Frequent Node:", analyzer.get_most_frequent_node())
                print("Longest Path Length:", analyzer.get_longest_path())
                print("Most Frequent Node Frequency:", analyzer.get_frequency(analyzer.get_most_frequent_node()))
                print("In-Degree of Most Frequent Node:", analyzer.get_in_degree(analyzer.get_most_frequent_node()))
                print("Out-Degree of Most Frequent Node:", analyzer.get_out_degree(analyzer.get_most_frequent_node()))
                print("\n" + "="*50 + "\n")

    # Sequential pattern mining
    print("\nFrequent Phase Patterns:")
    miner = SequentialPatternMiner(phase_sequences)
    phase_patterns = miner.find_frequent_patterns(min_support=0.3)
    for pattern in phase_patterns:
        print(pattern)

    most_freq = miner.get_most_frequent_patterns(phase_patterns)
    print("\nMost Frequent Phase Pattern(s):")
    for p, f in most_freq:
        print(f"{p}: {f}")

    longest = miner.get_longest_patterns(phase_patterns)
    print("\nLongest Phase Pattern(s):")
    for p, f in longest:
        print(f"{p}: {f}")