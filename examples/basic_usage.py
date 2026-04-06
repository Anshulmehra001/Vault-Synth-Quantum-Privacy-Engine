"""
Basic usage example for Vault-Synth

This script demonstrates the simplest way to use Vault-Synth
to generate synthetic financial data.
"""

import sys
sys.path.append('..')

from src.pipeline import VaultSynthPipeline

def main():
    """Run basic synthetic data generation"""
    
    print("Vault-Synth Basic Usage Example")
    print("=" * 60)
    
    # Create pipeline with default configuration
    pipeline = VaultSynthPipeline()
    
    # Run complete pipeline
    synthetic_data = pipeline.run(
        filepath='../data/sample_transactions.csv',
        n_synthetic_samples=50,  # Generate 50 synthetic samples
        output_path='../data/synthetic_output.csv'
    )
    
    # Display results
    print("\nSynthetic Data Preview:")
    print(synthetic_data.to_dataframe().head(10))
    
    print("\nSynthetic Data Statistics:")
    print(synthetic_data.to_dataframe().describe())
    
    print("\nDone! Synthetic data saved to '../data/synthetic_output.csv'")


if __name__ == "__main__":
    main()
