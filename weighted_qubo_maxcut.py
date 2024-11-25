from pyqubo import Binary
import neal
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt


G = nx.Graph()
edges_with_weights = [
    (1, 2, 0.5),
    (1, 3, 1.2),
    (2, 3, 0.8),
    (3, 4, 1.5),
    (4, 1, 0.9)
]
G.add_weighted_edges_from(edges_with_weights)


def formulate_qubo_weighted(graph):
    Q = defaultdict(int)
    for i, j, w in graph.edges(data="weight"):
        Q[(i, i)] -= w
        Q[(j, j)] -= w
        Q[(i, j)] += 2 * w
    return Q


def qubo_to_numpy(Q, num_nodes):
    numpy_Q = np.zeros((num_nodes, num_nodes))
    for (i, j), value in Q.items():
        numpy_Q[i - 1, j - 1] += value  # Node indices are 1-based in the graph
        if i != j:
            numpy_Q[j - 1, i - 1] += value  # Symmetric matrix
    return numpy_Q


def solve_maxcut_qubo(Q, num_reads=10):
    num_nodes = len(G.nodes)
    numpy_Q = qubo_to_numpy(Q, num_nodes)
    variables = [Binary(f"x{i}") for i in range(1, num_nodes + 1)]
    x_vect = np.array(variables)
    qubo_matrix = x_vect.T @ numpy_Q @ x_vect

    model = qubo_matrix.compile()
    bqm = model.to_bqm()

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample = min(decoded_samples, key=lambda x: x.energy)

    return best_sample.energy, best_sample.sample


Q = formulate_qubo_weighted(G)
energy, solution = solve_maxcut_qubo(Q)

print("Max-Cut Energy:", energy)
print("Solution:", solution)

color_map = ["orangered" if solution[f"x{node}"] == 0 else "royalblue" for node in G.nodes]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=2000, font_weight='bold')
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=10
)
plt.title("Max-Cut Weighted")
plt.show()
