import pennylane as qml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import jax; import optax


class FromQubo:
    def __init__(self, qubo_matrix: np.array, graph: nx.Graph) -> None:
        self.qubo_matrix = qubo_matrix
        self.graph = graph
        tol = 1e-8
        if self.qubo_matrix.shape[0] != self.qubo_matrix.shape[1]:
            raise ValueError("The QUBO matrix must be a square matrix!")
        if not np.allclose(self.qubo_matrix, self.qubo_matrix.T, atol=tol):
            raise ValueError("The QUBO matrix is not symmetric, please provide a symmetric matrix.")

    def create_quadratic_parts_from_qubo_matrix(self) -> list:
        quadratic_part = []
        triu_matrix = np.triu(self.qubo_matrix, k=1)
        edges = self.graph.edges()
        for i, j in edges:
            if i != j:
                quadratic_part.append((triu_matrix[i, j], qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)))
        '''for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if self.qubo_matrix[i, j] != 0:
                    quadratic_part.append((self.qubo_matrix[i, j], qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)))'''
        return quadratic_part

    def create_hamiltonian(self) -> qml.Hamiltonian:
        diagonal = np.diagonal(self.qubo_matrix)

        linear_part = [(coeff, qml.PauliZ(wires=i)) for i, coeff in enumerate(diagonal)]
        quadratic_part = self.create_quadratic_parts_from_qubo_matrix()

        coefficients = [term[0] for term in linear_part] + [term[0] for term in quadratic_part]
        ops = [term[1] for term in linear_part] + [term[1] for term in quadratic_part]
        return qml.Hamiltonian(coefficients, ops)


# Example adjacency matrix
qubo_matrix = np.array([
    [2, 1, 1, 0, 0],
    [1, 2, 0, 1, 0],
    [1, 0, 3, 1, 1],
    [0, 1, 1, 3, 1],
    [0, 0, 1, 1, 2]
])


diag = np.diag(np.diag(qubo_matrix))
adj_matrix = np.subtract(qubo_matrix, diag)
graph = nx.from_numpy_array(adj_matrix)

qubo = FromQubo(qubo_matrix, graph)
qubo_h = qubo.create_hamiltonian()
print(qubo_h)

dev = qml.device("default.qubit", wires=qubo_h.wires)


@qml.qnode(dev)
def qubo_problem(params):
    for p, w in zip(params, qubo_h.wires):
        qml.RY(p, wires=w)
    return qml.expval(qubo_h)


key = jax.random.PRNGKey(seed=12345)
params = jax.random.uniform(key, shape=(len(qubo_h.wires), ))

optax_optmizer = optax.adagrad(learning_rate=0.1)  ### Adagrad

opt_state = optax_optmizer.init(params)
steps = 100



for step in range(steps):
    grads = jax.grad(qubo_problem)(params)
    updates, opt_state = optax_optmizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(f"It {step + 1}:", qubo_problem(params))

trained_params = params

dev2 = qml.device('default.qubit', wires=qubo_h.wires, shots=1)
@qml.qnode(dev2)
def qubo_problem_sample(params):
    for p, w in zip(params, qubo_h.wires):
        qml.RY(p, wires=w)
    return qml.sample()


solution_bitstring = qubo_problem_sample(trained_params)
print("Solution found by QUBO: ", solution_bitstring)


nx.draw(graph, node_color=["r" if sol == 1 else "b" for sol in solution_bitstring], with_labels=True)
plt.show()