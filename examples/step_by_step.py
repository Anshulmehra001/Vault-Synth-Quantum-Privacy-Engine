"""
Step-by-step example for Vault-Synth

This script demonstrates how to use individual components
of the pipeline for more control.
"""

import sys
sys.path.append('..')

from src.config import VaultSynthConfig
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.quantum_circuit import QCBMCircuit
from src.training import TrainingLoop
from src.generator import SyntheticGenerator

def main():
    """Run step-by-step pipeline"""
    
    print("Vault-Synth Step-by-Step Example")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    loader = DataLoader(n_columns=6)
    data, column_names = loader.load_and_prepare('../data/sample_transactions.csv')
    print(f"Loaded {len(data)} samples")
    
    # Step 2: Preprocess
    print("\n[2/6] Preprocessing...")
    preprocessor = Preprocessor(n_bins=16)
    data_discrete, metadata = preprocessor.fit_transform(data)
    print(f"Discretized into {preprocessor.n_bins} bins")
    
    # Step 3: Build circuit
    print("\n[3/6] Building quantum circuit...")
    circuit = QCBMCircuit(n_qubits=6, n_layers=3)
    circuit.initialize_weights()
    print(f"Circuit with {circuit.n_qubits} qubits initialized")
    
    # Step 4: Train
    print("\n[4/6] Training circuit...")
    trainer = TrainingLoop(circuit, learning_rate=0.01)
    results = trainer.train(
        real_data=data_discrete,
        n_steps=50,  # Fewer steps for demo
        verbose=True
    )
    print(f"Training complete. Final loss: {results['final_loss']:.6f}")
    
    # Step 5: Generate
    print("\n[5/6] Generating synthetic data...")
    generator = SyntheticGenerator(circuit, preprocessor, column_names)
    synthetic_data = generator.generate(n_samples=20)
    print(f"Generated {synthetic_data.n_samples} samples")
    
    # Step 6: Validate
    print("\n[6/6] Validating...")
    privacy = generator.verify_privacy(
        synthetic_data.data.values,
        data.values
    )
    print(f"Privacy preserved: {privacy['privacy_preserved']}")
    
    stats = generator.compute_statistical_similarity(
        synthetic_data.data.values,
        data.values
    )
    print(f"Mean error: {stats['mean_relative_error'].mean():.4f}")
    
    # Display results
    print("\n" + "=" * 60)
    print("Synthetic Data Sample:")
    print(synthetic_data.to_dataframe().head())
    
    # Save
    synthetic_data.to_csv('../data/synthetic_stepbystep.csv')
    print("\nSaved to '../data/synthetic_stepbystep.csv'")


if __name__ == "__main__":
    main()
