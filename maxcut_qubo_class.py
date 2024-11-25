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


class FormulateQubo:
    def __init__(self, G: nx.Graph) -> None:
        self.G = G
        self.n = G.number_of_nodes()
        self.variables = [Binary(f"x{i}") for i in range(self.n)]

    def MakeQubo(self) -> defaultdict:
        Q = defaultdict(int)
        # method of arranging the QUBO matrix seen here: https://github.com/dwave-examples/maximum-cut/blob/master/maximum_cut.py
        for i, j in G.edges:
            Q[(i, i)] += -1
            Q[(j, j)] += -1
            Q[(i, j)] += 2

        return Q
    def _get_numpy_matrix(self) -> np.ndarray:

        Q = self.MakeQubo()
        numpy_Q_matrix = np.zeros((self.n, self.n))

        for key in Q.keys():
            if key[0] - 1 == key[1] - 1:
                numpy_Q_matrix[key[0] - 1, key[1] - 1] = Q[key]
            elif key[1] != key[0]:
                ## upper part
                numpy_Q_matrix[key[0] - 1, key[1] - 1] = Q[key] / 2
                ## lower part
                numpy_Q_matrix[key[1] - 1, key[0] - 1] = Q[key] / 2

        x_vect = np.array(self.variables)
        non_transpose_part = numpy_Q_matrix @ x_vect
        transpose_part = x_vect.T
        qubo_matrix_bin = transpose_part @ non_transpose_part

        return qubo_matrix_bin

    def Solver(self, num_reads: int) -> tuple:

        qubo_matrix_bin = self._get_numpy_matrix()
        model = qubo_matrix_bin.compile()
        bqm = model.to_bqm()
        sa = neal.SimulatedAnnealingSampler()
        sampleset = sa.sample(bqm, num_reads=num_reads)
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda x: x.energy)

        return best_sample.energy, sorted(best_sample.sample.items())


if __name__ == "__main__":

    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)])


    qubo_solver = FormulateQubo(G)

    energy, solution = qubo_solver.Solver(num_reads=10)


    print("Max cut energy:", energy)
    print("Solution:", solution)

    sampled_solution_dict = dict(solution)
    color_map = ['red' if sampled_solution_dict[key] == 0 else 'blue' for key in sampled_solution_dict.keys()]

    nx.draw(G, with_labels=True, node_color=color_map, font_weight='bold', node_size=500)
    plt.title("Max-Cut")
    plt.show()
