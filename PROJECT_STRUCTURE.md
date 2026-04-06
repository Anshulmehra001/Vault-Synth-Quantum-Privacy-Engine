# Vault-Synth Project Structure

```
vault-synth/
│
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── exceptions.py            # Custom exceptions
│   ├── data_loader.py           # CSV loading and validation
│   ├── preprocessor.py          # Data normalization and discretization
│   ├── quantum_circuit.py       # QCBM implementation (PennyLane)
│   ├── mmd_loss.py              # Maximum Mean Discrepancy loss
│   ├── training.py              # Training loop with Adam optimizer
│   ├── generator.py             # Synthetic data generation
│   └── pipeline.py              # End-to-end orchestration
│
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_basic.py            # Basic unit tests
│
├── examples/                     # Example scripts
│   ├── basic_usage.py           # Simple usage example
│   ├── custom_config.py         # Custom configuration example
│   └── step_by_step.py          # Detailed step-by-step example
│
├── data/                         # Data directory
│   ├── sample_transactions.csv  # Sample dataset
│   └── README.md                # Data format documentation
│
├── config/                       # Configuration files (empty for now)
│
├── main.py                       # Main entry point (CLI)
├── setup.py                      # Package setup
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
│
├── README.md                     # Main documentation
├── QUICKSTART.md                 # Quick start guide
├── LICENSE                       # MIT License
└── PROJECT_STRUCTURE.md          # This file

```

## Component Overview

### Core Components

1. **config.py**: Manages system configuration and hardware profiling
2. **data_loader.py**: Loads CSV files and validates data
3. **preprocessor.py**: Normalizes data to [0, π] and discretizes
4. **quantum_circuit.py**: Implements QCBM with PennyLane
5. **mmd_loss.py**: Computes Maximum Mean Discrepancy
6. **training.py**: Trains circuit using Adam optimizer
7. **generator.py**: Generates synthetic samples
8. **pipeline.py**: Orchestrates the complete workflow

### Entry Points

- **main.py**: Command-line interface
- **examples/**: Python scripts demonstrating usage
- **tests/**: Test suite

### Data Flow

```
CSV File → DataLoader → Preprocessor → Quantum Circuit → Training → Generator → Synthetic CSV
```

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| pipeline.py | Main orchestration | ~200 |
| quantum_circuit.py | QCBM implementation | ~150 |
| training.py | Training loop | ~180 |
| generator.py | Synthetic generation | ~170 |
| mmd_loss.py | Loss function | ~120 |
| preprocessor.py | Data preprocessing | ~130 |
| data_loader.py | Data loading | ~100 |
| config.py | Configuration | ~80 |

## Dependencies

- **PennyLane**: Quantum circuit simulation
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Scikit-learn**: Preprocessing utilities
- **psutil**: Hardware profiling
- **Hypothesis**: Property-based testing
- **Pytest**: Unit testing

## Usage Patterns

### Pattern 1: Quick Run
```bash
python main.py
```

### Pattern 2: Custom CLI
```bash
python main.py --input data.csv --samples 200 --qubits 8
```

### Pattern 3: Python API
```python
from src.pipeline import VaultSynthPipeline
pipeline = VaultSynthPipeline()
data = pipeline.run('data.csv', 100)
```

### Pattern 4: Step-by-Step
```python
from src.data_loader import DataLoader
from src.quantum_circuit import QCBMCircuit
# ... manual control of each component
```
