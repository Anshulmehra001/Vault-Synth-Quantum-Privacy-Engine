"""Data preprocessing for Vault-Synth"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class Preprocessor:
    """Preprocesses data for quantum circuit training"""
    
    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins
        self.metadata = {}
    
    def normalize_to_pi(self, data: pd.DataFrame) -> np.ndarray:
        """Normalize values to range [0, π]"""
        # Store min and max for each column
        self.metadata['min_values'] = data.min().values
        self.metadata['max_values'] = data.max().values
        
        # Normalize to [0, 1] then scale to [0, π]
        data_array = data.values
        normalized = (data_array - self.metadata['min_values']) / (
            self.metadata['max_values'] - self.metadata['min_values'] + 1e-10
        )
        
        # Scale to [0, π]
        normalized_pi = normalized * np.pi
        
        return normalized_pi
    
    def discretize(self, data: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins"""
        # Create bins from 0 to π
        bin_edges = np.linspace(0, np.pi, self.n_bins + 1)
        self.metadata['bin_edges'] = bin_edges
        
        # Digitize each column
        discretized = np.zeros_like(data, dtype=int)
        for col_idx in range(data.shape[1]):
            discretized[:, col_idx] = np.digitize(data[:, col_idx], bin_edges[:-1]) - 1
            # Clip to valid range [0, n_bins-1]
            discretized[:, col_idx] = np.clip(discretized[:, col_idx], 0, self.n_bins - 1)
        
        return discretized
    
    def inverse_transform(self, discretized_data: np.ndarray) -> np.ndarray:
        """Convert bins back to continuous values"""
        if 'bin_edges' not in self.metadata:
            raise ValueError("Preprocessor has not been fitted. Call normalize_to_pi and discretize first.")
        
        bin_edges = self.metadata['bin_edges']
        
        # Use bin centers for inverse transform
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Map discrete bins to continuous values
        continuous = np.zeros_like(discretized_data, dtype=float)
        for col_idx in range(discretized_data.shape[1]):
            continuous[:, col_idx] = bin_centers[discretized_data[:, col_idx]]
        
        # Denormalize from [0, π] back to original range
        denormalized = (continuous / np.pi) * (
            self.metadata['max_values'] - self.metadata['min_values']
        ) + self.metadata['min_values']
        
        return denormalized
    
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Fit preprocessor and transform data"""
        normalized = self.normalize_to_pi(data)
        discretized = self.discretize(normalized)
        
        return discretized, self.metadata
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        if not self.metadata:
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")
        
        # Use stored min/max for normalization
        data_array = data.values
        normalized = (data_array - self.metadata['min_values']) / (
            self.metadata['max_values'] - self.metadata['min_values'] + 1e-10
        )
        normalized_pi = normalized * np.pi
        
        # Discretize using stored bin edges
        discretized = np.zeros_like(normalized_pi, dtype=int)
        for col_idx in range(normalized_pi.shape[1]):
            discretized[:, col_idx] = np.digitize(
                normalized_pi[:, col_idx], 
                self.metadata['bin_edges'][:-1]
            ) - 1
            discretized[:, col_idx] = np.clip(discretized[:, col_idx], 0, self.n_bins - 1)
        
        return discretized
