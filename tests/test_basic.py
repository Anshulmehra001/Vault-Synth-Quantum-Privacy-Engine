"""Basic tests for Vault-Synth components"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from src.config import VaultSynthConfig, ConfigManager
from src.exceptions import InvalidQubitCountError


def test_config_validation():
    """Test configuration validation"""
    # Valid config
    config = VaultSynthConfig(n_qubits=6, n_layers=3)
    config.validate()
    
    # Invalid qubit count
    with pytest.raises(ValueError):
        config = VaultSynthConfig(n_qubits=20)
        config.validate()


def test_config_manager():
    """Test configuration manager"""
    manager = ConfigManager()
    config = manager.get_config()
    
    assert config.n_qubits == 6
    assert config.n_layers == 3
    
    # Update config
    manager.update_config(n_qubits=8)
    assert manager.get_config().n_qubits == 8


def test_hardware_check():
    """Test hardware constraint checking"""
    manager = ConfigManager()
    hw_check = manager.check_hardware_constraints()
    
    assert 'total_ram_gb' in hw_check
    assert 'available_ram_gb' in hw_check
    assert 'meets_requirements' in hw_check
    assert hw_check['total_ram_gb'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
