"""End-to-end pipeline for Vault-Synth"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from .config import ConfigManager, VaultSynthConfig
from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .quantum_circuit import QCBMCircuit
from .training import TrainingLoop
from .generator import SyntheticGenerator, SyntheticData
from .exceptions import HardwareConstraintError


class VaultSynthPipeline:
    """Complete pipeline for synthetic data generation"""
    
    def __init__(self, config: Optional[VaultSynthConfig] = None):
        self.config_manager = ConfigManager(config)
        self.config = self.config_manager.get_config()
        
        # Check hardware constraints
        hw_check = self.config_manager.check_hardware_constraints()
        if not hw_check['meets_requirements']:
            print(f"Warning: System has {hw_check['total_ram_gb']}GB RAM, "
                  f"recommended {hw_check['required_ram_gb']}GB")
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.circuit = None
        self.trainer = None
        self.generator = None
        
        # Store data
        self.real_data = None
        self.real_data_discrete = None
        self.column_names = None
    
    def load_data(self, filepath: str, column_names: Optional[list] = None) -> pd.DataFrame:
        """Load and prepare data"""
        print(f"Loading data from {filepath}...")
        
        self.data_loader = DataLoader(n_columns=self.config.n_columns)
        data, self.column_names = self.data_loader.load_and_prepare(filepath, column_names)
        
        self.real_data = data
        print(f"Loaded {len(data)} samples with {len(self.column_names)} columns")
        print(f"Columns: {self.column_names}")
        
        return data
    
    def preprocess_data(self) -> np.ndarray:
        """Preprocess loaded data"""
        if self.real_data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        print("Preprocessing data...")
        
        self.preprocessor = Preprocessor(n_bins=self.config.n_bins)
        self.real_data_discrete, metadata = self.preprocessor.fit_transform(self.real_data)
        
        print(f"Data normalized to [0, π] and discretized into {self.config.n_bins} bins")
        
        return self.real_data_discrete
    
    def build_circuit(self) -> QCBMCircuit:
        """Build quantum circuit"""
        print(f"Building quantum circuit with {self.config.n_qubits} qubits "
              f"and {self.config.n_layers} layers...")
        
        self.circuit = QCBMCircuit(
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers
        )
        self.circuit.initialize_weights()
        
        print("Circuit initialized successfully")
        
        return self.circuit
    
    def train_circuit(self, n_steps: Optional[int] = None, verbose: bool = True) -> Dict:
        """Train the quantum circuit"""
        if self.circuit is None:
            raise ValueError("Circuit not built. Call build_circuit first.")
        
        if self.real_data_discrete is None:
            raise ValueError("Data not preprocessed. Call preprocess_data first.")
        
        n_steps = n_steps or self.config.n_training_steps
        
        print(f"\nTraining circuit for {n_steps} steps...")
        print(f"Learning rate: {self.config.learning_rate}")
        
        self.trainer = TrainingLoop(
            circuit=self.circuit,
            learning_rate=self.config.learning_rate
        )
        
        results = self.trainer.train(
            real_data=self.real_data_discrete,
            n_steps=n_steps,
            verbose=verbose
        )
        
        print(f"\nTraining complete!")
        print(f"Final loss: {results['final_loss']:.6f}")
        print(f"Converged: {results['converged']}")
        
        return results
    
    def generate_synthetic_data(self, n_samples: int) -> SyntheticData:
        """Generate synthetic data"""
        if self.circuit is None or self.circuit.weights is None:
            raise ValueError("Circuit not trained. Call train_circuit first.")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized. Call preprocess_data first.")
        
        print(f"\nGenerating {n_samples} synthetic samples...")
        
        self.generator = SyntheticGenerator(
            circuit=self.circuit,
            preprocessor=self.preprocessor,
            column_names=self.column_names
        )
        
        synthetic_data = self.generator.generate(n_samples)
        
        print(f"Generated {n_samples} synthetic samples")
        
        return synthetic_data
    
    def validate_privacy(self, synthetic_data: SyntheticData) -> Dict:
        """Validate privacy guarantees"""
        print("\nValidating privacy guarantees...")
        
        privacy_results = self.generator.verify_privacy(
            synthetic_data.data.values,
            self.real_data.values
        )
        
        print(f"Privacy preserved: {privacy_results['privacy_preserved']}")
        print(f"Exact matches: {privacy_results['n_exact_matches']}")
        
        return privacy_results
    
    def validate_statistical_similarity(self, synthetic_data: SyntheticData) -> Dict:
        """Validate statistical similarity"""
        print("\nValidating statistical similarity...")
        
        stats = self.generator.compute_statistical_similarity(
            synthetic_data.data.values,
            self.real_data.values
        )
        
        print("Mean relative error:", np.mean(stats['mean_relative_error']))
        print("Variance relative error:", np.mean(stats['variance_relative_error']))
        
        return stats
    
    def run(
        self,
        filepath: str,
        n_synthetic_samples: int,
        column_names: Optional[list] = None,
        n_training_steps: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> SyntheticData:
        """
        Run complete pipeline
        
        Args:
            filepath: Path to input CSV file
            n_synthetic_samples: Number of synthetic samples to generate
            column_names: Optional list of column names to use
            n_training_steps: Optional number of training steps
            output_path: Optional path to save synthetic data
        
        Returns:
            SyntheticData object
        """
        print("=" * 60)
        print("VAULT-SYNTH: Quantum Privacy Engine")
        print("=" * 60)
        
        # Load data
        self.load_data(filepath, column_names)
        
        # Preprocess
        self.preprocess_data()
        
        # Build circuit
        self.build_circuit()
        
        # Train
        self.train_circuit(n_steps=n_training_steps)
        
        # Generate
        synthetic_data = self.generate_synthetic_data(n_synthetic_samples)
        
        # Validate
        self.validate_privacy(synthetic_data)
        self.validate_statistical_similarity(synthetic_data)
        
        # Save if requested
        if output_path:
            synthetic_data.to_csv(output_path)
            print(f"\nSynthetic data saved to {output_path}")
        
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        
        return synthetic_data
