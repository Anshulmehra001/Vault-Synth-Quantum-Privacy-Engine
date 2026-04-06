# Sample Dataset

## sample_transactions.csv

This is a sample credit card transaction dataset for testing Vault-Synth.

### Format

- **Time**: Time elapsed since first transaction (in seconds)
- **V1-V5**: Principal components from PCA transformation (anonymized features)
- **Amount**: Transaction amount in dollars
- **Class**: 0 = legitimate transaction, 1 = fraudulent transaction

### Source

This is a simplified version inspired by the Credit Card Fraud Detection dataset.
The data has been anonymized using PCA to protect privacy.

### Usage

```python
from src.pipeline import VaultSynthPipeline

pipeline = VaultSynthPipeline()
synthetic_data = pipeline.run(
    filepath='data/sample_transactions.csv',
    n_synthetic_samples=100
)
```

### Note

For production use, you should use a larger dataset (at least 1000+ samples) for better statistical learning.
