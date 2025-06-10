from gsppy.gsp import GSP

# Example transactions: customer purchases
transactions = [
    ['Bread', 'Milk'],  # Transaction 1
    ['Bread', 'Diaper', 'Beer', 'Bread', 'Eggs'],  # Transaction 2
    ['Milk', 'Diaper', 'Beer', 'Coke'],  # Transaction 3
    ['Bread', 'Milk', 'Diaper', 'Beer'],  # Transaction 4
    ['Bread', 'Milk', 'Diaper', 'Coke']  # Transaction 5
]

# Set minimum support threshold (30%)
min_support = 0.3

# Find frequent patterns
result = GSP(transactions).search(min_support)

# Output the results
print(result)

import networkx as nx

G = nx.DiGraph()
edges = [("A", "B"), ("B", "C"), ("C", "B"), ("B", "D"), ("D", "B"), ("B", "E"), ("E", "F"), ("F", "G"), ("G", "B")]
G.add_edges_from(edges)

# Find strongly connected components (SCCs)
sccs = list(nx.strongly_connected_components(G))
print("Strongly Connected Components:", sccs)

# Collapse each SCC into a supernode
condensed = nx.condensation(G, sccs)
nx.set_node_attributes(condensed, {i: scc for i, scc in enumerate(sccs)}, "members")

# Print the condensed graph
print("Condensed Graph Nodes:", condensed.nodes(data=True))
print("Longest Path in Condensed Graph:", nx.dag_longest_path(condensed))
