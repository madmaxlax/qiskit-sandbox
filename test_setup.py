#!/usr/bin/env python3
"""
Test script to verify Qiskit setup is working correctly
"""

import sys
import importlib


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")

    packages = [
        "qiskit",
        "qiskit.quantum_info",
        "qiskit.circuit.library",
        "qiskit.primitives",
        "matplotlib",
        "numpy",
        "scipy",
        "rustworkx",
    ]

    failed_imports = []

    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_basic_qiskit():
    """Test basic Qiskit functionality"""
    print("\nTesting basic Qiskit functionality...")

    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator

        # Create a simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Create an observable
        observable = SparsePauliOp("ZZ")

        # Run estimation
        estimator = StatevectorEstimator()
        job = estimator.run([(qc, observable)])
        result = job.result()

        expectation_value = result[0].data.evs
        print(f"✓ Bell state expectation value: {expectation_value}")

        return True

    except Exception as e:
        print(f"✗ Basic Qiskit test failed: {e}")
        return False


def test_visualization():
    """Test visualization capabilities"""
    print("\nTesting visualization capabilities...")

    try:
        import matplotlib.pyplot as plt
        from qiskit import QuantumCircuit

        # Create a simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Test circuit drawing
        circuit_diagram = qc.draw(output="text")
        print("✓ Circuit drawing works")

        # Test matplotlib
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        print("✓ Matplotlib plotting works")

        return True

    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Qiskit Sandbox Setup Test ===\n")

    # Test imports
    imports_ok = test_imports()

    # Test basic functionality
    qiskit_ok = test_basic_qiskit()

    # Test visualization
    viz_ok = test_visualization()

    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    if all([imports_ok, qiskit_ok, viz_ok]):
        print("🎉 All tests passed! Your Qiskit sandbox is ready to use.")
        print("\nYou can now run:")
        print("  python hello_world_example.py")
        print("  python qaoa_example.py")
        return 0
    else:
        print("❌ Some tests failed. Please check your setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
