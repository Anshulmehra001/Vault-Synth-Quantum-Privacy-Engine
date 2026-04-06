"""Configuration management for Vault-Synth"""

import psutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class VaultSynthConfig:
    """Configuration for the Vault-Synth quantum privacy engine"""
    
    # Quantum circuit parameters
    n_qubits: int = 6
    n_layers: int = 3
    
    # Training parameters
    n_training_steps: int = 200
    learning_rate: float = 0.01
    
    # Data parameters
    n_columns: int = 6
    n_bins: int = 16
    
    # Hardware constraints
    max_ram_gb: float = 16.0
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if not (4 <= self.n_qubits <= 12):
            raise ValueError(f"n_qubits must be between 4 and 12, got {self.n_qubits}")
        
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {self.n_layers}")
        
        if self.n_training_steps < 1:
            raise ValueError(f"n_training_steps must be at least 1, got {self.n_training_steps}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.n_columns < 1:
            raise ValueError(f"n_columns must be at least 1, got {self.n_columns}")
        
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {self.n_bins}")


class ConfigManager:
    """Manages configuration and hardware profiling"""
    
    def __init__(self, config: Optional[VaultSynthConfig] = None):
        self.config = config or VaultSynthConfig()
        self.config.validate()
    
    def check_hardware_constraints(self) -> dict:
        """Check if hardware meets requirements"""
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        meets_requirements = total_ram_gb >= self.config.max_ram_gb
        
        return {
            "total_ram_gb": round(total_ram_gb, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "required_ram_gb": self.config.max_ram_gb,
            "meets_requirements": meets_requirements,
        }
    
    def get_config(self) -> VaultSynthConfig:
        """Get the current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        self.config.validate()
