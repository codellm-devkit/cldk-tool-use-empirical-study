from collections import defaultdict, Counter

def aggregate_repeated_ngrams(sequences, min_count=2, max_n=4):
    pattern_counter = Counter()
    for seq in sequences:
        local_seen = set()
        for n in range(2, max_n + 1):
            for i in range(len(seq) - n + 1):
                pat = tuple(seq[i:i+n])
                if seq.count(pat) >= min_count:
                    local_seen.add(pat)
        for pat in local_seen:
            pattern_counter[pat] += 1
    return pattern_counter

sequences = [
    ['A', 'B', 'C', 'A', 'B', 'C', 'D'],
    ['A', 'B', 'C', 'A', 'B', 'C', 'D'],
    ['A', 'B', 'C', 'A', 'B', 'C'],
    ['A', 'B', 'C', 'D'],
    ['A', 'B', 'C']
]
if __name__ == "__main__":
    result = aggregate_repeated_ngrams(sequences, min_count=2, max_n=4)
    print("Aggregated Repeated n-grams:", result)
# import getpass

# print(getpass.getuser())

# from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
# difficulty_levels = []
# for item in ds:
#     if item["difficulty"] not in difficulty_levels:
#         difficulty_levels.append(item["difficulty"])
# print("Difficulty Levels:", difficulty_levels)
# from gsppy.gsp import GSP

# # Example transactions: customer purchases
# transactions = [
#     ['Bread', 'Milk'],  # Transaction 1
#     ['Bread', 'Diaper', 'Beer', 'Bread', 'Eggs'],  # Transaction 2
#     ['Milk', 'Diaper', 'Beer', 'Coke'],  # Transaction 3
#     ['Bread', 'Milk', 'Diaper', 'Beer'],  # Transaction 4
#     ['Bread', 'Milk', 'Diaper', 'Coke']  # Transaction 5
# ]

# # Set minimum support threshold (30%)
# min_support = 0.3

# # Find frequent patterns
# result = GSP(transactions).search(min_support)

# # Output the results
# print(result)

# import networkx as nx

# G = nx.DiGraph()
# edges = [("A", "B"), ("B", "C"), ("C", "B"), ("B", "D"), ("D", "B"), ("B", "E"), ("E", "F"), ("F", "G"), ("G", "B")]
# G.add_edges_from(edges)

# # Find strongly connected components (SCCs)
# sccs = list(nx.strongly_connected_components(G))
# print("Strongly Connected Components:", sccs)

# # Collapse each SCC into a supernode
# condensed = nx.condensation(G, sccs)
# nx.set_node_attributes(condensed, {i: scc for i, scc in enumerate(sccs)}, "members")

# # Print the condensed graph
# print("Condensed Graph Nodes:", condensed.nodes(data=True))
# print("Longest Path in Condensed Graph:", len(nx.dag_longest_path(condensed)))
