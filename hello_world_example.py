#!/usr/bin/env python3
"""
Qiskit Hello World Example
Based on: https://quantum.cloud.ibm.com/docs/en/tutorials/hello-world

This example demonstrates the four steps to writing a quantum program using Qiskit patterns:
1. Map the problem to a quantum-native format
2. Optimize the circuits and operators
3. Execute using a quantum primitive function
4. Analyze the results

This creates a Bell state - a state where two qubits are fully entangled with each other.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

# For cloud execution (uncomment when ready to use IBM Cloud)
# from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
# from qiskit_ibm_runtime import EstimatorOptions

# For local simulation
from qiskit.primitives import StatevectorEstimator as LocalEstimator


def create_bell_state_circuit():
    """
    Step 1: Map the problem to a quantum-native format
    Create a circuit that produces a Bell state
    """
    print("Step 1: Creating Bell state circuit...")

    # Create a new circuit with two qubits
    qc = QuantumCircuit(2)

    # Add a Hadamard gate to qubit 0
    qc.h(0)

    # Perform a controlled-X gate on qubit 1, controlled by qubit 0
    qc.cx(0, 1)

    print("Bell state circuit created!")
    print(qc.draw(output="text"))

    return qc


def create_observables():
    """
    Create observables (operators) to measure expectation values
    """
    print("\nCreating observables...")

    # Set up six different observables
    observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
    observables = [SparsePauliOp(label) for label in observables_labels]

    print(f"Created {len(observables)} observables: {observables_labels}")

    return observables, observables_labels


def optimize_circuit_local(qc):
    """
    Step 2: Optimize the circuits and operators (local simulation version)
    """
    print("\nStep 2: Optimizing circuit for local simulation...")

    # For local simulation, we don't need backend-specific optimization
    # but we can still apply basic optimization
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # Create a basic pass manager for optimization
    pm = generate_preset_pass_manager(optimization_level=1)
    optimized_circuit = pm.run(qc)

    print("Circuit optimized for local simulation")
    print(f"Original circuit depth: {qc.depth()}")
    print(f"Optimized circuit depth: {optimized_circuit.depth()}")

    return optimized_circuit


def execute_local(circuit, observables):
    """
    Step 3: Execute using local simulation
    """
    print("\nStep 3: Executing on local simulator...")

    # Use local estimator for simulation
    estimator = LocalEstimator()

    # Run the estimation
    job = estimator.run([(circuit, obs) for obs in observables])
    result = job.result()

    # Extract expectation values
    expectation_values = [pubres.data.evs for pubres in result]

    print("Execution completed on local simulator")

    return expectation_values


def execute_cloud_placeholder(circuit, observables):
    """
    Step 3: Execute using IBM Cloud (placeholder - requires authentication)

    To use this, you need to:
    1. Set up IBM Cloud account
    2. Save your credentials using QiskitRuntimeService.save_account()
    3. Uncomment the imports at the top
    4. Uncomment the code below
    """
    print("\nStep 3: IBM Cloud execution (placeholder)...")
    print("To execute on IBM Cloud hardware:")
    print("1. Set up IBM Cloud account")
    print(
        "2. Save credentials: QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_TOKEN')"
    )
    print("3. Uncomment the cloud execution code")

    # Uncomment and modify this code when ready for cloud execution:
    """
    # Initialize service and get backend
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    
    # Optimize circuit for the specific backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    
    # Map observables to the backend
    isa_observables = [obs.apply_layout(isa_circuit.layout) for obs in observables]
    
    # Set up estimator with error mitigation
    estimator = Estimator(backend)
    options = EstimatorOptions()
    options.resilience_level = 1
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options = options
    
    # Execute
    job = estimator.run([(isa_circuit, obs) for obs in isa_observables])
    result = job.result()
    
    return [pubres.data.evs for pubres in result]
    """

    # Return dummy values for demonstration
    return [np.array([0.0]) for _ in observables]


def analyze_results(expectation_values, observables_labels, use_cloud=False):
    """
    Step 4: Analyze the results
    """
    print(
        f"\nStep 4: Analyzing results ({'IBM Cloud' if use_cloud else 'Local Simulation'})..."
    )

    print("\nExpectation values:")
    for label, value in zip(observables_labels, expectation_values):
        # Handle different data structures from different estimators
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            try:
                # For arrays/lists
                if hasattr(value, "__len__") and len(value) > 0:
                    val = value[0]
                else:
                    val = 0.0
            except (TypeError, IndexError):
                # Fallback for unsized iterables
                val = float(value) if hasattr(value, "__float__") else 0.0
        else:
            # For scalar values
            val = value
        print(f"⟨{label}⟩ = {val:.6f}")

    # For a perfect Bell state |00⟩ + |11⟩, we expect:
    # ⟨ZZ⟩ = 1 (both qubits have the same Z measurement)
    # ⟨IZ⟩ = ⟨ZI⟩ = 0 (individual Z measurements average to 0)
    # ⟨XX⟩ = 1 (both qubits have correlated X measurements)
    # ⟨IX⟩ = ⟨XI⟩ = 0 (individual X measurements average to 0)

    print("\nFor a perfect Bell state, we expect:")
    print("⟨ZZ⟩ ≈ 1 (qubits are correlated in Z basis)")
    print("⟨XX⟩ ≈ 1 (qubits are correlated in X basis)")
    print("⟨IZ⟩, ⟨ZI⟩, ⟨IX⟩, ⟨XI⟩ ≈ 0 (individual qubit measurements)")


def main():
    """
    Main function demonstrating the complete Hello World workflow
    """
    print("=== Qiskit Hello World Example ===")
    print("Creating and analyzing a Bell state using Qiskit patterns\n")

    # Step 1: Create the Bell state circuit
    circuit = create_bell_state_circuit()

    # Create observables to measure
    observables, observables_labels = create_observables()

    # Step 2: Optimize the circuit
    optimized_circuit = optimize_circuit_local(circuit)

    # Step 3 & 4: Execute and analyze (local simulation)
    print("\n" + "=" * 50)
    print("RUNNING LOCAL SIMULATION")
    print("=" * 50)

    expectation_values_local = execute_local(optimized_circuit, observables)
    analyze_results(expectation_values_local, observables_labels, use_cloud=False)

    # Step 3 & 4: Execute and analyze (cloud placeholder)
    print("\n" + "=" * 50)
    print("IBM CLOUD EXECUTION (PLACEHOLDER)")
    print("=" * 50)

    expectation_values_cloud = execute_cloud_placeholder(optimized_circuit, observables)
    analyze_results(expectation_values_cloud, observables_labels, use_cloud=True)

    print("\n=== Example completed! ===")
    print("\nNext steps:")
    print("1. Set up IBM Cloud account for real quantum hardware")
    print("2. Try the QAOA example for more complex quantum algorithms")
    print("3. Explore other Qiskit tutorials and examples")


if __name__ == "__main__":
    main()
