"""Quantum Circuit Born Machine implementation"""

import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .exceptions import InvalidQubitCountError, UntrainedCircuitError


@dataclass
class CircuitParameters:
    """Parameters for the quantum circuit"""
    weights: np.ndarray
    n_qubits: int
    n_layers: int
    
    def validate(self):
        """Validate parameter shapes"""
        expected_shape = (self.n_layers, self.n_qubits, 3)
        if self.weights.shape != expected_shape:
            raise ValueError(
                f"Weight shape {self.weights.shape} does not match "
                f"expected shape {expected_shape}"
            )


class QCBMCircuit:
    """Quantum Circuit Born Machine for learning probability distributions"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        if not (4 <= n_qubits <= 12):
            raise InvalidQubitCountError(
                f"n_qubits must be between 4 and 12, got {n_qubits}"
            )
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.weights = None
        self._circuit_func = None
        
        # Build the circuit
        self._build_circuit()
    
    def _build_circuit(self):
        """Build the quantum circuit with Hadamard gates and StronglyEntanglingLayers"""
        @qml.qnode(self.device)
        def circuit(weights):
            # Apply Hadamard gates to all qubits for superposition
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            
            # Apply StronglyEntanglingLayers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Return probability distribution
            return qml.probs(wires=range(self.n_qubits))
        
        self._circuit_func = circuit
    
    def initialize_weights(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize random weights for the circuit"""
        if seed is not None:
            np.random.seed(seed)
        
        # Shape: (n_layers, n_qubits, 3) for StronglyEntanglingLayers
        self.weights = np.random.uniform(
            low=0, 
            high=2 * np.pi, 
            size=(self.n_layers, self.n_qubits, 3)
        )
        
        return self.weights
    
    def get_probabilities(self, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Get probability distribution from the circuit"""
        if weights is None:
            if self.weights is None:
                raise UntrainedCircuitError(
                    "Circuit has not been initialized. Call initialize_weights first."
                )
            weights = self.weights
        
        # Validate weights
        params = CircuitParameters(weights, self.n_qubits, self.n_layers)
        params.validate()
        
        probs = self._circuit_func(weights)
        return probs
    
    def sample(self, n_samples: int = 1, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate samples from the circuit's probability distribution"""
        probs = self.get_probabilities(weights)
        
        # Sample from the probability distribution
        indices = np.random.choice(
            len(probs), 
            size=n_samples, 
            p=probs
        )
        
        return indices
    
    def set_weights(self, weights: np.ndarray):
        """Set circuit weights"""
        params = CircuitParameters(weights, self.n_qubits, self.n_layers)
        params.validate()
        self.weights = weights
    
    def get_circuit_func(self):
        """Get the underlying circuit function for training"""
        return self._circuit_func
