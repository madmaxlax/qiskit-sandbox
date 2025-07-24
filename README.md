# Qiskit Sandbox

A playground repository for learning and experimenting with IBM's Qiskit quantum computing SDK. This sandbox includes working examples that demonstrate quantum computing concepts and algorithms, with both local simulation and IBM Cloud quantum hardware execution capabilities.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+ (Recommended: Python 3.12.7 for best compatibility with Qiskit 2.1+)
- pip package manager
- pyenv (optional, for Python version management)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd qiskit-sandbox
   ```

2. **Create and activate virtual environment**
   ```bash
   # If using pyenv, set Python version first:
   # pyenv local 3.12.7
   
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install "qiskit[all]~=2.1.1" "qiskit-ibm-runtime~=0.40.1" matplotlib jupyter rustworkx
   ```

4. **Run examples**
   ```bash
   # Hello World example (Bell state)
   python hello_world_example.py
   
   # QAOA optimization example
   python qaoa_example.py
   ```

## 📚 Examples

### 1. Hello World Example (`hello_world_example.py`)

**What it does:** Creates and analyzes a Bell state - a fundamental quantum state where two qubits are fully entangled.

**Key concepts demonstrated:**
- Quantum circuit creation and manipulation
- Bell state generation (|00⟩ + |11⟩)
- Expectation value measurements
- Local simulation vs. cloud execution
- Qiskit's four-step pattern:
  1. Map problem to quantum-native format
  2. Optimize circuits and operators
  3. Execute using quantum primitives
  4. Analyze results

**Expected output:**
- ⟨ZZ⟩ ≈ 1 (qubits correlated in Z basis)
- ⟨XX⟩ ≈ 1 (qubits correlated in X basis)
- ⟨IZ⟩, ⟨ZI⟩, ⟨IX⟩, ⟨XI⟩ ≈ 0 (individual measurements)

**Files generated:**
- Console output showing expectation values
- Placeholder for IBM Cloud execution

### 2. QAOA Example (`qaoa_example.py`)

**What it does:** Implements the Quantum Approximate Optimization Algorithm (QAOA) to solve the Maximum-Cut (Max-Cut) problem.

**Key concepts demonstrated:**
- Combinatorial optimization problems
- Hybrid quantum-classical algorithms
- Graph-to-Hamiltonian mapping
- Parameter optimization
- Quantum sampling and result analysis

**Problem:** Given a graph, partition nodes into two sets to maximize the number of edges between the sets.

**Expected output:**
- Graph visualization (`max_cut_graph.png`)
- Optimization progress plot (`qaoa_optimization.png`)
- Solution visualization (`max_cut_solution.png`)
- Best solution bitstring and cut value

**Files generated:**
- `max_cut_graph.png` - Input graph visualization
- `qaoa_optimization.png` - Optimization convergence plot
- `max_cut_solution.png` - Solution visualization (red/blue nodes)

## 🔧 IBM Cloud Integration

Both examples include placeholders for IBM Cloud quantum hardware execution. To use real quantum hardware:

### Setup IBM Cloud Account

1. **Create IBM Cloud account**
   - Visit [IBM Quantum Platform](https://quantum.ibm.com/)
   - Sign up for a free account

2. **Get API token**
   - Go to your account settings
   - Generate an API token

3. **Configure credentials**
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   
   # Save your credentials
   QiskitRuntimeService.save_account(
       channel='ibm_cloud',
       token='YOUR_API_TOKEN'
   )
   ```

4. **Uncomment cloud execution code**
   - Edit the example files
   - Uncomment the IBM Cloud imports and execution code
   - Replace placeholder values with your configuration

### Cloud Execution Features

- **Error mitigation** with dynamical decoupling
- **Resilience levels** for noise suppression
- **Session-based execution** for optimization loops
- **Hardware-specific transpilation** and optimization

## 🛠️ Development

### Project Structure

```
qiskit-sandbox/
├── README.md                 # This file
├── hello_world_example.py    # Bell state example
├── qaoa_example.py          # QAOA optimization example
├── venv/                    # Virtual environment
└── *.png                    # Generated visualizations
```

### Dependencies

- **qiskit[all]~=2.1.1** - Core Qiskit framework
- **qiskit-ibm-runtime~=0.40.1** - IBM Cloud integration
- **matplotlib** - Plotting and visualization
- **jupyter** - Interactive notebooks (optional)
- **rustworkx** - Graph operations and visualization
- **scipy** - Scientific computing and optimization

### Adding New Examples

1. Create a new Python file following the naming convention
2. Implement the four-step Qiskit pattern:
   - Map problem to quantum format
   - Optimize for execution
   - Execute using primitives
   - Analyze results
3. Include both local simulation and cloud execution options
4. Add comprehensive documentation and comments

## 📖 Learning Resources

- [IBM Quantum Learning](https://learning.quantum.ibm.com/) - Official tutorials
- [Qiskit Documentation](https://qiskit.org/documentation/) - API reference
- [Qiskit Textbook](https://qiskit.org/textbook/) - Comprehensive learning material
- [IBM Quantum Platform](https://quantum.ibm.com/) - Cloud quantum computing

## 🎯 Next Steps

1. **Explore more algorithms:**
   - Grover's algorithm for search
   - Variational Quantum Eigensolver (VQE)
   - Quantum Fourier Transform (QFT)

2. **Try different problems:**
   - Traveling Salesman Problem
   - Portfolio optimization
   - Machine learning applications

3. **Experiment with hardware:**
   - Run on different IBM quantum devices
   - Compare simulator vs. hardware results
   - Explore error mitigation techniques

4. **Build your own applications:**
   - Custom quantum circuits
   - Hybrid quantum-classical workflows
   - Real-world optimization problems

## 🤝 Contributing

Feel free to:
- Add new examples and algorithms
- Improve existing code and documentation
- Report issues or suggest enhancements
- Share your quantum computing experiments

## 📄 License

This project is for educational purposes. Please refer to the Qiskit license for framework usage.

---

**Happy quantum computing! 🚀⚛️**
