# Vault-Synth: Quantum Privacy Engine

A quantum-inspired synthetic data generator using Quantum Circuit Born Machines (QCBM) to create privacy-preserving financial datasets for AI training and GDPR compliance.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44-orange)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

Vault-Synth explores quantum machine learning techniques for generating synthetic financial data that preserves statistical properties while ensuring privacy compliance. Built as a research and learning project, it demonstrates how Quantum Circuit Born Machines can be applied to the growing challenge of GDPR-compliant data sharing.

**Key Features:**
- 🔬 Quantum Circuit Born Machine (QCBM) implementation using PennyLane
- 🔐 Privacy-preserving synthetic data generation
- 📊 Statistical similarity preservation (mean, variance, correlations)
- 🎓 Educational codebase for quantum machine learning
- 💻 Runs on classical simulators (no quantum hardware required)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Anshulmehra001/Vault-Synth-Quantum-Privacy-Engine.git
cd Vault-Synth-Quantum-Privacy-Engine

# Install dependencies
pip install pennylane numpy pandas scikit-learn
```

### Run the Application

```bash
# Generate 100 synthetic samples with default settings
python main.py

# Custom configuration
python main.py --samples 200 --qubits 8 --steps 300
```

### Python API Usage

```python
from src.pipeline import VaultSynthPipeline

# Create pipeline with default configuration
pipeline = VaultSynthPipeline()

# Generate synthetic data
synthetic_data = pipeline.run(
    filepath='data/sample_transactions.csv',
    n_synthetic_samples=100,
    output_path='synthetic_output.csv'
)

# Access the generated data
df = synthetic_data.to_dataframe()
print(df.head())
```

## 🏗️ Architecture

```
Input CSV → Data Preprocessing → Quantum Circuit → Training → Synthetic Generation
    ↓            ↓                    ↓              ↓              ↓
Real Data    Normalize to [0,π]   6-10 Qubits   MMD Loss    Privacy-Safe Output
             Discretize           QCBM Circuit   Optimizer   (No Real Data)
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Loader** | Pandas | CSV loading and validation |
| **Preprocessor** | NumPy, Scikit-learn | Normalization and discretization |
| **Quantum Circuit** | PennyLane | QCBM with StronglyEntanglingLayers |
| **Training Engine** | PennyLane Adam | MMD loss optimization |
| **Generator** | NumPy | Synthetic sample generation |

## 📊 How It Works

### 1. Data Preprocessing
- Load financial transaction data (CSV format)
- Normalize numerical values to range [0, π]
- Discretize continuous values into bins for quantum processing

### 2. Quantum Circuit Training
- Initialize 6-10 qubit circuit with random parameters
- Apply Hadamard gates for quantum superposition
- Use StronglyEntanglingLayers to learn data correlations
- Optimize using Maximum Mean Discrepancy (MMD) loss

### 3. Synthetic Data Generation
- Sample from trained quantum probability distribution
- Convert discrete quantum states back to continuous values
- Denormalize to original data range

### 4. Privacy Validation
- Verify no exact matches with original data
- Compare statistical properties (mean, variance)
- Ensure generated samples are novel

## 🔧 Configuration

Customize the quantum circuit and training parameters:

```python
from src.config import VaultSynthConfig
from src.pipeline import VaultSynthPipeline

config = VaultSynthConfig(
    n_qubits=8,              # Number of qubits (4-12)
    n_layers=3,              # Circuit depth
    n_training_steps=200,    # Training iterations
    learning_rate=0.01,      # Adam optimizer learning rate
    n_bins=16                # Discretization bins
)

pipeline = VaultSynthPipeline(config)
```

## 📈 Example Output

```
============================================================
VAULT-SYNTH: Quantum Privacy Engine
============================================================
Loading data from data/sample_transactions.csv...
Loaded 42 samples with 6 columns

Building quantum circuit with 6 qubits and 3 layers...
Training circuit for 200 steps...
Step 20/200, Loss: 0.234567
Step 40/200, Loss: 0.198234
...
Training complete! Final loss: 0.559491

Generating 100 synthetic samples...
Privacy preserved: True (0 exact matches)

Statistical Similarity:
  Mean relative error: 0.77
  Variance relative error: 1.55

Synthetic data saved to synthetic_output.csv
============================================================
```

## 🧪 Project Structure

```
vault-synth/
├── src/                      # Source code
│   ├── config.py            # Configuration management
│   ├── data_loader.py       # CSV loading and validation
│   ├── preprocessor.py      # Data normalization
│   ├── quantum_circuit.py   # QCBM implementation
│   ├── mmd_loss.py          # MMD loss function
│   ├── training.py          # Training loop
│   ├── generator.py         # Synthetic data generation
│   └── pipeline.py          # End-to-end orchestration
├── examples/                 # Usage examples
│   ├── basic_usage.py
│   ├── custom_config.py
│   └── step_by_step.py
├── data/                     # Sample datasets
│   └── sample_transactions.csv
├── tests/                    # Test suite
├── main.py                   # CLI entry point
└── README.md
```

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.9+ |
| RAM | 8GB | 16GB |
| CPU | Any modern CPU | Multi-core |
| OS | Windows/Mac/Linux | Any |

**Note:** No quantum hardware required - runs on classical simulators.

## 🎓 Research Context

This project explores Quantum Circuit Born Machines (QCBMs) for generative modeling, a research area in quantum machine learning. While quantum advantage for this application is not yet realized on current NISQ (Noisy Intermediate-Scale Quantum) devices, this implementation demonstrates:

- Quantum circuit design for probability distribution learning
- Integration of quantum computing frameworks with classical ML pipelines
- Privacy-preserving data generation techniques
- Practical application of quantum-inspired algorithms

**Current Status:** Research/Educational project. Classical methods (GANs, VAEs) currently outperform QCBMs for production use cases, but quantum approaches show theoretical promise for future quantum hardware.

## 📚 Use Cases

### Educational
- Learn quantum machine learning concepts
- Understand QCBM architecture and training
- Explore privacy-preserving AI techniques

### Research
- Experiment with quantum generative models
- Compare quantum vs classical approaches
- Develop new quantum circuit architectures

### Prototyping
- Test synthetic data generation workflows
- Evaluate privacy preservation methods
- Demonstrate quantum computing concepts

## 🔬 Technical Details

### Quantum Circuit Born Machine (QCBM)

A QCBM is a quantum generative model that learns probability distributions using parameterized quantum circuits:
- Leverages quantum superposition to represent multiple states
- Uses entanglement to capture complex data correlations
- Outputs probability distributions natively through measurement

### Maximum Mean Discrepancy (MMD)

MMD measures statistical distance between distributions:
- Uses Gaussian kernel for smooth comparison
- Minimized during training to match real data distribution
- Ensures statistical similarity without memorization

### Implementation Details

- **Framework**: PennyLane (quantum ML library)
- **Backend**: Classical simulator (default.qubit)
- **Optimizer**: Adam with configurable learning rate
- **Circuit Depth**: 3-4 layers of StronglyEntanglingLayers
- **Qubit Range**: 4-12 qubits (configurable)

## 📖 References & Resources

### Academic Papers
- [Quantum Circuit Born Machines](https://arxiv.org/abs/1804.04168) - Original QCBM paper
- [Generative Quantum Learning](https://arxiv.org/abs/1904.09557) - Quantum generative models overview

### Industry Resources
- [PennyLane Documentation](https://pennylane.ai/)
- [Quantum Machine Learning](https://pennylane.ai/qml/)
- [Synthetic Data for GDPR Compliance](https://gdpr.eu/)

### Related Work
- Gartner: 60% of AI training data will be synthetic by 2024
- Financial institutions exploring quantum ML (JPMorgan, Goldman Sachs)
- Active research in quantum advantage for generative modeling

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Enhanced circuit architectures
- Better convergence strategies
- Additional privacy metrics
- Performance optimizations
- Extended documentation

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is a research and educational project exploring quantum machine learning concepts. For production synthetic data generation, consider established classical methods (GANs, VAEs) which currently offer better performance and reliability. Always consult with legal and privacy experts for GDPR compliance in production environments.

## 🙏 Acknowledgments

- Built with [PennyLane](https://pennylane.ai/) quantum computing framework
- Inspired by research in quantum generative modeling
- Sample data format based on credit card fraud detection datasets

## 📧 Contact

**Author:** Anshul Mehra  
**GitHub:** [@Anshulmehra001](https://github.com/Anshulmehra001)  
**Project:** [Vault-Synth-Quantum-Privacy-Engine](https://github.com/Anshulmehra001/Vault-Synth-Quantum-Privacy-Engine)

---

⭐ If you find this project interesting, please consider giving it a star!

**Note:** This project demonstrates quantum ML concepts for educational purposes. Quantum advantage for synthetic data generation is an active research area, with practical applications expected in the coming years as quantum hardware improves.
