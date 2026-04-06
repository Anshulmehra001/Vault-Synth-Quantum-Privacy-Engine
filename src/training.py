"""Training loop for Quantum Circuit Born Machine"""

import numpy as np
import pennylane as qml
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from .quantum_circuit import QCBMCircuit
from .mmd_loss import MMDLoss
from .exceptions import ConvergenceError


@dataclass
class TrainingState:
    """Tracks training state"""
    current_step: int = 0
    loss_history: List[float] = field(default_factory=list)
    best_loss: float = float('inf')
    best_weights: Optional[np.ndarray] = None
    converged: bool = False


class TrainingLoop:
    """Trains the QCBM using MMD loss and Adam optimizer"""
    
    def __init__(
        self,
        circuit: QCBMCircuit,
        learning_rate: float = 0.01,
        convergence_threshold: float = 1e-4,
        patience: int = 20
    ):
        self.circuit = circuit
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        
        self.mmd_loss = MMDLoss(sigma=1.0)
        self.state = TrainingState()
        
        # Initialize optimizer
        self.optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    
    def compute_loss(self, weights: np.ndarray, real_data: np.ndarray) -> float:
        """
        Compute MMD loss between real data and quantum-generated samples
        
        Args:
            weights: Circuit parameters
            real_data: Real data samples (n_samples, n_features)
        
        Returns:
            MMD loss value
        """
        # Get probability distribution from circuit
        probs = self.circuit.get_probabilities(weights)
        
        # Generate synthetic samples from the circuit
        n_samples = len(real_data)
        synthetic_indices = np.random.choice(
            len(probs),
            size=n_samples,
            p=probs
        )
        
        # Convert indices to binary representations (features)
        n_qubits = self.circuit.n_qubits
        synthetic_data = np.array([
            [int(b) for b in format(idx, f'0{n_qubits}b')]
            for idx in synthetic_indices
        ])
        
        # Compute MMD loss
        loss = self.mmd_loss(real_data, synthetic_data, vectorized=True)
        
        return loss
    
    def check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.state.loss_history) < self.patience:
            return False
        
        recent_losses = self.state.loss_history[-self.patience:]
        loss_std = np.std(recent_losses)
        
        if loss_std < self.convergence_threshold:
            self.state.converged = True
            return True
        
        return False
    
    def train_step(self, weights: np.ndarray, real_data: np.ndarray) -> tuple:
        """
        Perform one training step
        
        Args:
            weights: Current circuit parameters
            real_data: Real data samples
        
        Returns:
            (updated_weights, loss)
        """
        # Define cost function for optimizer
        def cost_fn(w):
            return self.compute_loss(w, real_data)
        
        # Perform gradient descent step
        updated_weights, loss = self.optimizer.step_and_cost(cost_fn, weights)
        
        return updated_weights, loss
    
    def train(
        self,
        real_data: np.ndarray,
        n_steps: int = 200,
        verbose: bool = True,
        early_stopping: bool = True
    ) -> dict:
        """
        Train the quantum circuit
        
        Args:
            real_data: Real data samples (n_samples, n_features)
            n_steps: Number of training steps
            verbose: Print progress
            early_stopping: Stop early if converged
        
        Returns:
            Training results dictionary
        """
        # Initialize weights if not already done
        if self.circuit.weights is None:
            self.circuit.initialize_weights()
        
        weights = self.circuit.weights.copy()
        
        # Reset training state
        self.state = TrainingState()
        
        for step in range(n_steps):
            # Perform training step
            weights, loss = self.train_step(weights, real_data)
            
            # Update state
            self.state.current_step = step + 1
            self.state.loss_history.append(float(loss))
            
            # Track best weights
            if loss < self.state.best_loss:
                self.state.best_loss = float(loss)
                self.state.best_weights = weights.copy()
            
            # Print progress
            if verbose and (step + 1) % 20 == 0:
                print(f"Step {step + 1}/{n_steps}, Loss: {loss:.6f}")
            
            # Check convergence
            if early_stopping and self.check_convergence():
                if verbose:
                    print(f"Converged at step {step + 1}")
                break
        
        # Set circuit to best weights
        if self.state.best_weights is not None:
            self.circuit.set_weights(self.state.best_weights)
        else:
            self.circuit.set_weights(weights)
        
        # Check if training was successful
        if self.state.best_loss > 0.5:  # Threshold for acceptable loss
            if verbose:
                print(f"Warning: Training may not have converged well. Final loss: {self.state.best_loss:.6f}")
        
        return {
            'final_loss': self.state.best_loss,
            'loss_history': self.state.loss_history,
            'n_steps': self.state.current_step,
            'converged': self.state.converged,
            'best_weights': self.state.best_weights
        }
