"""Synthetic data generation module"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from .quantum_circuit import QCBMCircuit
from .preprocessor import Preprocessor
from .exceptions import UntrainedCircuitError


@dataclass
class SyntheticData:
    """Container for synthetic data with metadata"""
    data: pd.DataFrame
    n_samples: int
    timestamp: str
    column_names: list
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return data as DataFrame"""
        return self.data
    
    def to_csv(self, filepath: str):
        """Save synthetic data to CSV"""
        self.data.to_csv(filepath, index=False)


class SyntheticGenerator:
    """Generates synthetic data from trained quantum circuit"""
    
    def __init__(
        self,
        circuit: QCBMCircuit,
        preprocessor: Preprocessor,
        column_names: Optional[list] = None
    ):
        self.circuit = circuit
        self.preprocessor = preprocessor
        self.column_names = column_names or [f"feature_{i}" for i in range(circuit.n_qubits)]
    
    def generate(self, n_samples: int, batch_size: int = 100) -> SyntheticData:
        """
        Generate synthetic samples
        
        Args:
            n_samples: Number of samples to generate
            batch_size: Batch size for generation (for memory efficiency)
        
        Returns:
            SyntheticData object
        """
        if self.circuit.weights is None:
            raise UntrainedCircuitError(
                "Circuit has not been trained. Train the circuit before generating data."
            )
        
        # Generate samples in batches
        all_samples = []
        remaining = n_samples
        
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batch_samples = self._generate_batch(current_batch)
            all_samples.append(batch_samples)
            remaining -= current_batch
        
        # Concatenate all batches
        synthetic_discrete = np.vstack(all_samples)
        
        # Convert from discrete bins back to continuous values
        synthetic_continuous = self.preprocessor.inverse_transform(synthetic_discrete)
        
        # Create DataFrame
        df = pd.DataFrame(
            synthetic_continuous,
            columns=self.column_names[:synthetic_continuous.shape[1]]
        )
        
        # Create SyntheticData object
        synthetic_data = SyntheticData(
            data=df,
            n_samples=n_samples,
            timestamp=datetime.now().isoformat(),
            column_names=self.column_names[:synthetic_continuous.shape[1]]
        )
        
        return synthetic_data
    
    def _generate_batch(self, batch_size: int) -> np.ndarray:
        """
        Generate a batch of synthetic samples
        
        Args:
            batch_size: Number of samples in batch
        
        Returns:
            Discrete samples (batch_size, n_features)
        """
        # Get probability distribution from circuit
        probs = self.circuit.get_probabilities()
        
        # Sample indices from the distribution
        indices = np.random.choice(
            len(probs),
            size=batch_size,
            p=probs
        )
        
        # Convert indices to binary representations (discrete features)
        n_qubits = self.circuit.n_qubits
        samples = np.array([
            [int(b) for b in format(idx, f'0{n_qubits}b')]
            for idx in indices
        ])
        
        # Map binary values to bin indices
        # Binary values are 0 or 1, we need to map them to bin indices
        # For simplicity, we'll scale them to the number of bins
        n_bins = self.preprocessor.n_bins
        samples_scaled = (samples * (n_bins - 1)).astype(int)
        
        return samples_scaled
    
    def verify_privacy(self, synthetic_data: np.ndarray, real_data: np.ndarray, tolerance: float = 1e-6) -> dict:
        """
        Verify that synthetic data doesn't contain exact copies of real data
        
        Args:
            synthetic_data: Generated synthetic data
            real_data: Original real data
            tolerance: Tolerance for floating point comparison
        
        Returns:
            Dictionary with privacy verification results
        """
        n_exact_matches = 0
        
        for synth_row in synthetic_data:
            for real_row in real_data:
                if np.allclose(synth_row, real_row, atol=tolerance):
                    n_exact_matches += 1
                    break
        
        privacy_preserved = (n_exact_matches == 0)
        
        return {
            'privacy_preserved': privacy_preserved,
            'n_exact_matches': n_exact_matches,
            'match_percentage': (n_exact_matches / len(synthetic_data)) * 100
        }
    
    def compute_statistical_similarity(self, synthetic_data: np.ndarray, real_data: np.ndarray) -> dict:
        """
        Compare statistical properties of synthetic and real data
        
        Args:
            synthetic_data: Generated synthetic data
            real_data: Original real data
        
        Returns:
            Dictionary with statistical comparison metrics
        """
        metrics = {}
        
        # Mean comparison
        real_mean = np.mean(real_data, axis=0)
        synth_mean = np.mean(synthetic_data, axis=0)
        mean_diff = np.abs(real_mean - synth_mean)
        
        # Variance comparison
        real_var = np.var(real_data, axis=0)
        synth_var = np.var(synthetic_data, axis=0)
        var_diff = np.abs(real_var - synth_var)
        
        metrics['real_mean'] = real_mean
        metrics['synthetic_mean'] = synth_mean
        metrics['mean_difference'] = mean_diff
        metrics['mean_relative_error'] = mean_diff / (np.abs(real_mean) + 1e-10)
        
        metrics['real_variance'] = real_var
        metrics['synthetic_variance'] = synth_var
        metrics['variance_difference'] = var_diff
        metrics['variance_relative_error'] = var_diff / (real_var + 1e-10)
        
        return metrics
