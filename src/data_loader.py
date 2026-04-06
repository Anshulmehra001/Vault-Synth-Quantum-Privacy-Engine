"""Data loading and validation for Vault-Synth"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from .exceptions import InvalidDataError, InsufficientDataError


class DataLoader:
    """Loads and validates CSV data for quantum circuit training"""
    
    def __init__(self, n_columns: int = 6):
        self.n_columns = n_columns
        self.data = None
        self.selected_columns = None
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV file into memory"""
        try:
            path = Path(filepath)
            if not path.exists():
                raise InvalidDataError(f"File not found: {filepath}")
            
            self.data = pd.read_csv(filepath)
            
            if self.data.empty:
                raise InsufficientDataError("CSV file is empty")
            
            return self.data
        
        except pd.errors.EmptyDataError:
            raise InvalidDataError(f"CSV file is empty or corrupted: {filepath}")
        except pd.errors.ParserError as e:
            raise InvalidDataError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            raise InvalidDataError(f"Error loading CSV file: {e}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Check for NaN and infinite values"""
        if data.isnull().any().any():
            raise InvalidDataError("Data contains NaN values")
        
        if np.isinf(data.select_dtypes(include=[np.number]).values).any():
            raise InvalidDataError("Data contains infinite values")
        
        return True
    
    def select_columns(self, data: pd.DataFrame, column_names: list = None) -> pd.DataFrame:
        """Select numerical columns from the dataset"""
        # Get all numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < self.n_columns:
            raise InsufficientDataError(
                f"Dataset has only {len(numerical_cols)} numerical columns, "
                f"but {self.n_columns} are required"
            )
        
        # Use specified columns or select first n_columns
        if column_names:
            if not all(col in numerical_cols for col in column_names):
                raise InvalidDataError("Some specified columns are not numerical or don't exist")
            selected = data[column_names[:self.n_columns]]
        else:
            selected = data[numerical_cols[:self.n_columns]]
        
        self.selected_columns = selected.columns.tolist()
        return selected
    
    def load_and_prepare(self, filepath: str, column_names: list = None) -> Tuple[pd.DataFrame, list]:
        """Load CSV, validate, and select columns"""
        data = self.load_csv(filepath)
        selected = self.select_columns(data, column_names)
        self.validate_data(selected)
        
        return selected, self.selected_columns
