from pyqubo import Binary
import neal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def MakeSimpleGraph(n: int, prob: float, seed: int) -> nx.Graph:
    """
    Generate an Erdos-Rényi graph with n nodes and a probability of conneting the nodes as prob.
    :param n: (int) Node of the graph;
    :param prob: (float) Probability of connecting node to each other;
    :param seed: (int) Seed for reproducibility;
    :return: G (nx.Graph) The Erdos-Rényi graph.
    """

    assert prob >= 0 and prob <= 1

    G = nx.fast_gnp_random_graph(n, prob, seed)
    return G


n = 5; p=0.8; seed=8888

x1, x2, x3, x4, x5 = Binary('x1'), Binary('x2'), Binary('x3'), Binary('x4'), Binary('x5')

## formulate as qubo

#G = MakeSimpleGraph(n, p, seed)

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)])

'''
nx.draw(G, with_labels=True)
plt.show()
'''


Q = defaultdict(int)


# method of arranging the QUBO matrix seen here: https://github.com/dwave-examples/maximum-cut/blob/master/maximum_cut.py
for i, j in G.edges:
    Q[(i, i)] += -1
    Q[(j, j)] += -1
    Q[(i, j)] += 2

print("Keys:", list(Q.keys()))
print("Values:", [Q[key] for key in Q.keys()])


numpy_Q_matrix = np.zeros((n, n)) # I create the numpy qubo matrix placing the coeffs

for key in Q.keys():
    if key[0] - 1 == key[1] - 1:
        numpy_Q_matrix[key[0] - 1, key[1] - 1] = Q[key]
    elif key[1] != key[0]:
        ## upper part
        numpy_Q_matrix[key[0] - 1, key[1] - 1] = Q[key]/2
        ## lower part
        numpy_Q_matrix[key[1] - 1, key[0] - 1] = Q[key]/2


print(numpy_Q_matrix)

x_vect = np.array([x1, x2, x3, x4, x5])

non_transpose_part = numpy_Q_matrix @ x_vect
transpose_part = x_vect.T
qubo_matrix_bin = transpose_part @ non_transpose_part

#print(qubo_matrix_bin.compile().to_qubo())
#### PyQubo documentation commands
model = qubo_matrix_bin.compile()
bqm = model.to_bqm()

sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=10)
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)


print("Max cut:", best_sample.energy)
sampled_solution = best_sample.sample

color_map = ['red' if sampled_solution[key] == 0 else 'blue' for key in sampled_solution.keys()]

nx.draw(G, with_labels=True, node_color=color_map, font_weight='bold', node_size = 500)
plt.show()

