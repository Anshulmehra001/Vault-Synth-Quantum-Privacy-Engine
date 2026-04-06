"""
Custom configuration example for Vault-Synth

This script shows how to customize the quantum circuit
and training parameters.
"""

import sys
sys.path.append('..')

from src.pipeline import VaultSynthPipeline
from src.config import VaultSynthConfig

def main():
    """Run with custom configuration"""
    
    print("Vault-Synth Custom Configuration Example")
    print("=" * 60)
    
    # Create custom configuration
    config = VaultSynthConfig(
        n_qubits=8,              # Use 8 qubits (more expressive)
        n_layers=4,              # Use 4 layers (deeper circuit)
        n_training_steps=100,    # Fewer steps for faster training
        learning_rate=0.02,      # Higher learning rate
        n_columns=6,             # Use 6 columns from data
        n_bins=32                # More bins for finer discretization
    )
    
    print("\nConfiguration:")
    print(f"  Qubits: {config.n_qubits}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Training steps: {config.n_training_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Bins: {config.n_bins}")
    
    # Create pipeline with custom config
    pipeline = VaultSynthPipeline(config)
    
    # Run pipeline
    synthetic_data = pipeline.run(
        filepath='../data/sample_transactions.csv',
        n_synthetic_samples=100,
        output_path='../data/synthetic_custom.csv'
    )
    
    print("\nSynthetic Data Shape:", synthetic_data.data.shape)
    print("\nFirst 5 samples:")
    print(synthetic_data.to_dataframe().head())


if __name__ == "__main__":
    main()
