#!/usr/bin/env python3
"""
Quantum Approximate Optimization Algorithm (QAOA) Example
Based on: https://learning.quantum.ibm.com/tutorial/quantum-approximate-optimization-algorithm

This example demonstrates QAOA for solving the Maximum-Cut (Max-Cut) problem:
- Given a graph, partition nodes into two sets to maximize edges between sets
- This is an NP-hard combinatorial optimization problem
- QAOA is a hybrid quantum-classical algorithm that can find approximate solutions

The four steps of Qiskit patterns:
1. Map classical problem to quantum circuits and operators
2. Optimize circuits for quantum hardware execution
3. Execute using quantum primitives
4. Analyze and post-process results
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# For cloud execution (uncomment when ready)
# from qiskit_ibm_runtime import QiskitRuntimeService, Session
# from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
# from qiskit_ibm_runtime import EstimatorOptions

# For local simulation
from qiskit.primitives import StatevectorEstimator as LocalEstimator
from qiskit.primitives import StatevectorSampler as LocalSampler

# Graph visualization (rustworkx is the recommended library)
try:
    import rustworkx as rx
    from rustworkx.visualization import mpl_draw as draw_graph

    RX_AVAILABLE = True
except ImportError:
    print("rustworkx not available. Install with: pip install rustworkx")
    RX_AVAILABLE = False


def create_sample_graph():
    """
    Create a sample graph for the Max-Cut problem
    """
    print("Step 1: Creating sample graph...")

    # Create a 5-node graph
    n = 5

    if RX_AVAILABLE:
        graph = rx.PyGraph()
        graph.add_nodes_from(np.arange(0, n, 1))
        edge_list = [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 4, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
        ]
        graph.add_edges_from(edge_list)

        print(
            f"Created graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges"
        )

        # Visualize if matplotlib is available
        try:
            draw_graph(graph, node_size=600, with_labels=True)
            plt.title("Sample Graph for Max-Cut Problem")
            plt.savefig("max_cut_graph.png")
            plt.close()
            print("Graph visualization saved as 'max_cut_graph.png'")
        except Exception as e:
            print(f"Could not create graph visualization: {e}")

        return graph
    else:
        # Fallback: represent as adjacency list
        print("Using adjacency list representation (rustworkx not available)")
        edges = [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3), (3, 4)]
        return {"n": n, "edges": edges}


def build_max_cut_hamiltonian(graph):
    """
    Convert graph to cost Hamiltonian for QAOA

    For Max-Cut, the cost Hamiltonian is:
    H_C = sum_{(i,j) in edges} (1/2)(1 - Z_i Z_j)

    This gives +1 when qubits i and j are in different states (edge in cut)
    and 0 when they are in the same state (edge not in cut)
    """
    print("Converting graph to quantum Hamiltonian...")

    if RX_AVAILABLE and hasattr(graph, "edge_list"):
        # Using rustworkx graph
        n = len(graph.nodes())
        edges = list(graph.edge_list())
    else:
        # Using adjacency list
        n = graph["n"]
        edges = graph["edges"]

    pauli_list = []
    for edge in edges:
        # For each edge (i,j), add Z_i Z_j term
        pauli_str = ["I"] * n
        pauli_str[edge[0]] = "Z"
        pauli_str[edge[1]] = "Z"
        pauli_list.append(("".join(pauli_str), 1.0))

    cost_hamiltonian = SparsePauliOp.from_list(pauli_list)

    print(f"Created cost Hamiltonian with {len(pauli_list)} terms")
    print(f"Hamiltonian: {cost_hamiltonian}")

    return cost_hamiltonian, n


def create_qaoa_circuit(cost_hamiltonian, reps=1, add_measurements=True):
    """
    Create QAOA circuit with specified number of repetitions
    """
    print(f"Creating QAOA circuit with {reps} repetitions...")

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

    if add_measurements:
        circuit.measure_all()

    print(f"QAOA circuit created with {circuit.num_parameters} parameters")
    print(f"Circuit depth: {circuit.depth()}")

    return circuit


def cost_function_local(params, ansatz, hamiltonian):
    """
    Cost function for local optimization
    """
    # Bind parameters to circuit
    bound_circuit = ansatz.assign_parameters(params)

    # Use local estimator
    estimator = LocalEstimator()

    # Compute expectation value
    job = estimator.run([(bound_circuit, hamiltonian)])
    result = job.result()

    cost = result[0].data.evs
    return cost


def optimize_qaoa_local(circuit, hamiltonian):
    """
    Optimize QAOA parameters using local simulation
    """
    print("\nStep 3: Optimizing QAOA parameters (local simulation)...")

    # Initial parameters
    num_params = circuit.num_parameters
    initial_params = np.random.uniform(0, 2 * np.pi, num_params)

    print(f"Starting optimization with {num_params} parameters...")

    # Track objective function values
    objective_values = []

    def cost_wrapper(params):
        cost = cost_function_local(params, circuit, hamiltonian)
        objective_values.append(cost)
        return cost

    # Optimize using scipy
    result = minimize(
        cost_wrapper, initial_params, method="COBYLA", options={"maxiter": 100}
    )

    print(f"Optimization completed!")
    print(f"Final cost: {result.fun:.6f}")
    print(f"Optimization steps: {len(objective_values)}")

    # Plot optimization progress
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(objective_values)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("QAOA Optimization Progress")
        plt.grid(True)
        plt.savefig("qaoa_optimization.png")
        plt.close()
        print("Optimization plot saved as 'qaoa_optimization.png'")
    except Exception as e:
        print(f"Could not create optimization plot: {e}")

    return result.x, objective_values


def sample_qaoa_solution(circuit, optimal_params, shots=10000):
    """
    Sample from the optimized QAOA circuit
    """
    print(f"\nStep 4: Sampling solutions ({shots} shots)...")

    # Bind optimal parameters
    optimized_circuit = circuit.assign_parameters(optimal_params)

    # Sample using local sampler
    sampler = LocalSampler()
    job = sampler.run([optimized_circuit], shots=shots)
    result = job.result()

    # Get counts
    counts = result[0].data.meas.get_counts()

    print(f"Sampling completed with {len(counts)} unique bitstrings")

    return counts


def analyze_max_cut_results(counts, graph):
    """
    Analyze the Max-Cut results
    """
    print("\nAnalyzing Max-Cut results...")

    # Find the most likely bitstring
    most_likely = max(counts, key=counts.get)
    probability = counts[most_likely] / sum(counts.values())

    print(f"Most likely solution: {most_likely}")
    print(f"Probability: {probability:.4f}")

    # Calculate cut value
    cut_value = evaluate_cut(most_likely, graph)
    print(f"Cut value: {cut_value}")

    # Show top solutions
    print("\nTop 5 solutions:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for i, (bitstring, count) in enumerate(sorted_counts[:5]):
        prob = count / sum(counts.values())
        cut_val = evaluate_cut(bitstring, graph)
        print(f"{i+1}. {bitstring} (prob: {prob:.4f}, cut: {cut_val})")

    return most_likely, cut_value


def evaluate_cut(bitstring, graph):
    """
    Evaluate the cut value for a given bitstring solution
    """
    if RX_AVAILABLE and hasattr(graph, "edge_list"):
        edges = list(graph.edge_list())
    else:
        edges = graph["edges"]

    # Convert bitstring to list of integers
    solution = [int(bit) for bit in bitstring]

    # Count edges in the cut
    cut_value = 0
    for edge in edges:
        i, j = edge[0], edge[1]
        if solution[i] != solution[j]:  # Nodes in different partitions
            cut_value += 1

    return cut_value


def cloud_execution_placeholder():
    """
    Placeholder for IBM Cloud execution
    """
    print("\n" + "=" * 50)
    print("IBM CLOUD EXECUTION (PLACEHOLDER)")
    print("=" * 50)

    print("To run QAOA on IBM quantum hardware:")
    print("1. Set up IBM Cloud account")
    print("2. Save credentials using QiskitRuntimeService.save_account()")
    print("3. Uncomment cloud execution imports")
    print("4. Use the following pattern:")

    print(
        """
# Example cloud execution code:
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True, min_num_qubits=100)

# Transpile for hardware
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
isa_circuit = pm.run(circuit)

# Run optimization in a session
with Session(backend=backend) as session:
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000
    estimator.options.resilience_level = 1
    estimator.options.dynamical_decoupling.enable = True
    
    # Run optimization...
    result = minimize(cost_function, initial_params, ...)
    
    # Sample final solution
    sampler = Sampler(mode=session)
    job = sampler.run([optimized_circuit], shots=10000)
    """
    )


def main():
    """
    Main QAOA example workflow
    """
    print("=== QAOA (Quantum Approximate Optimization Algorithm) Example ===")
    print("Solving Maximum-Cut problem using quantum-classical hybrid optimization\n")

    # Step 1: Create problem and map to quantum
    graph = create_sample_graph()
    cost_hamiltonian, n = build_max_cut_hamiltonian(graph)

    # Step 2: Create and optimize QAOA circuit
    # Create circuit without measurements for optimization
    qaoa_circuit = create_qaoa_circuit(cost_hamiltonian, reps=2, add_measurements=False)

    print("\nStep 2: Circuit created and ready for optimization")
    print(f"Problem size: {n} qubits")
    print(f"QAOA layers: 2")

    # Step 3: Optimize parameters (local simulation)
    print("\n" + "=" * 50)
    print("RUNNING LOCAL SIMULATION")
    print("=" * 50)

    optimal_params, objective_values = optimize_qaoa_local(
        qaoa_circuit, cost_hamiltonian
    )

    # Step 4: Sample and analyze results
    # Create circuit with measurements for sampling
    qaoa_circuit_with_measurements = create_qaoa_circuit(
        cost_hamiltonian, reps=2, add_measurements=True
    )
    counts = sample_qaoa_solution(qaoa_circuit_with_measurements, optimal_params)
    best_solution, best_cut_value = analyze_max_cut_results(counts, graph)

    # Visualize solution
    try:
        visualize_cut_solution(graph, best_solution)
    except Exception as e:
        print(f"Could not create solution visualization: {e}")

    # Show cloud execution information
    cloud_execution_placeholder()

    print("\n=== QAOA Example completed! ===")
    print(f"Best solution found: {best_solution}")
    print(f"Maximum cut value: {best_cut_value}")
    print("\nNext steps:")
    print("1. Try different graph structures and sizes")
    print("2. Experiment with different numbers of QAOA layers")
    print("3. Run on real quantum hardware using IBM Cloud")


def visualize_cut_solution(graph, solution):
    """
    Visualize the Max-Cut solution
    """
    if not RX_AVAILABLE:
        print("Cannot visualize solution without rustworkx")
        return

    try:
        # Create colors based on solution
        colors = ["red" if int(bit) == 0 else "blue" for bit in solution]

        # Draw graph with colored nodes
        draw_graph(graph, node_color=colors, node_size=600, with_labels=True)
        plt.title(f"Max-Cut Solution: {solution}")
        plt.figtext(
            0.1, 0.02, "Red nodes: partition 0, Blue nodes: partition 1", fontsize=10
        )
        plt.savefig("max_cut_solution.png")
        plt.close()
        print("Solution visualization saved as 'max_cut_solution.png'")

    except Exception as e:
        print(f"Could not create solution visualization: {e}")


if __name__ == "__main__":
    main()
