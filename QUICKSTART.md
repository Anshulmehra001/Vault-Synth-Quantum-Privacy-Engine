# Vault-Synth Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Option 1: Run with Default Settings

```bash
python main.py
```

This will:
- Load sample data from `data/sample_transactions.csv`
- Train a 6-qubit quantum circuit
- Generate 100 synthetic samples
- Save to `data/synthetic_output.csv`

### Option 2: Custom Parameters

```bash
python main.py --input data/sample_transactions.csv \
               --output my_synthetic_data.csv \
               --samples 200 \
               --qubits 8 \
               --steps 150
```

### Option 3: Use as Python Library

```python
from src.pipeline import VaultSynthPipeline

# Create pipeline
pipeline = VaultSynthPipeline()

# Generate synthetic data
synthetic_data = pipeline.run(
    filepath='data/sample_transactions.csv',
    n_synthetic_samples=100
)

# Access the data
df = synthetic_data.to_dataframe()
print(df.head())
```

## 📊 What You'll See

```
============================================================
VAULT-SYNTH: Quantum Privacy Engine
============================================================
Loading data from data/sample_transactions.csv...
Loaded 31 samples with 6 columns

Building quantum circuit with 6 qubits and 3 layers...
Training circuit for 200 steps...
Step 20/200, Loss: 0.234567
...
Training complete!

Generating 100 synthetic samples...
Privacy preserved: True
Exact matches: 0

Pipeline complete!
============================================================
```

## 🎯 Next Steps

1. **Try the examples**:
   ```bash
   cd examples
   python basic_usage.py
   python custom_config.py
   python step_by_step.py
   ```

2. **Use your own data**:
   - Replace `data/sample_transactions.csv` with your CSV file
   - Make sure it has numerical columns
   - The system will automatically select the first 6 columns

3. **Experiment with parameters**:
   - More qubits (8-10) = better for complex data
   - More training steps (300-500) = better convergence
   - Higher learning rate (0.02-0.05) = faster but less stable

## ⚠️ Important Notes

- **No downloads needed**: All code is ready to run (just install dependencies)
- **Dependencies**: Run `pip install -r requirements.txt` first
- **Hardware**: Works on any laptop with 8GB+ RAM
- **Time**: Training takes 5-15 minutes depending on settings

## 🔧 Troubleshooting

**Issue**: "Module not found"
```bash
pip install -r requirements.txt
```

**Issue**: "Out of memory"
- Reduce `n_qubits` to 4 or 5
- Reduce dataset size (use first 100 rows)

**Issue**: "Training not converging"
- Increase `n_training_steps` to 300
- Decrease `learning_rate` to 0.005

## 📚 Learn More

- Read the full [README.md](README.md)
- Check out [examples/](examples/) for more use cases
- See [data/README.md](data/README.md) for data format details
