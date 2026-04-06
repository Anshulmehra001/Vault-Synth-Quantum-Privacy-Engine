"""
Main entry point for Vault-Synth

Run this script to generate synthetic data with default settings.
"""

from src.pipeline import VaultSynthPipeline
from src.config import VaultSynthConfig
import argparse


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Vault-Synth: Quantum Privacy Engine for Synthetic Data Generation'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/sample_transactions.csv',
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/synthetic_output.csv',
        help='Path to save synthetic data'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of synthetic samples to generate'
    )
    
    parser.add_argument(
        '--qubits',
        type=int,
        default=6,
        help='Number of qubits (4-12)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help='Number of training steps'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate for optimizer'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = VaultSynthConfig(
        n_qubits=args.qubits,
        n_training_steps=args.steps,
        learning_rate=args.learning_rate
    )
    
    # Create and run pipeline
    pipeline = VaultSynthPipeline(config)
    
    synthetic_data = pipeline.run(
        filepath=args.input,
        n_synthetic_samples=args.samples,
        output_path=args.output
    )
    
    print(f"\n✅ Success! Generated {args.samples} synthetic samples")
    print(f"📁 Saved to: {args.output}")


if __name__ == "__main__":
    main()
