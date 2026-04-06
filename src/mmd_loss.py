"""Maximum Mean Discrepancy loss implementation"""

import numpy as np


class MMDLoss:
    """Computes Maximum Mean Discrepancy between two distributions"""
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize MMD loss with Gaussian kernel
        
        Args:
            sigma: Bandwidth parameter for Gaussian kernel
        """
        self.sigma = sigma
    
    def gaussian_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Gaussian (RBF) kernel between two vectors
        
        K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        """
        diff = x - y
        squared_dist = np.sum(diff ** 2)
        return np.exp(-squared_dist / (2 * self.sigma ** 2))
    
    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute kernel matrix between two sets of samples
        
        Args:
            X: First set of samples (n_samples_x, n_features)
            Y: Second set of samples (n_samples_y, n_features)
        
        Returns:
            Average kernel value
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]
        
        kernel_sum = 0.0
        for i in range(n_x):
            for j in range(n_y):
                kernel_sum += self.gaussian_kernel(X[i], Y[j])
        
        return kernel_sum / (n_x * n_y)
    
    def compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Maximum Mean Discrepancy between two distributions
        
        MMD^2(X, Y) = E[K(x, x')] - 2*E[K(x, y)] + E[K(y, y')]
        
        Args:
            X: Samples from first distribution (n_samples_x, n_features)
            Y: Samples from second distribution (n_samples_y, n_features)
        
        Returns:
            MMD value (always non-negative)
        """
        # Ensure inputs are 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Compute kernel expectations
        k_xx = self.compute_kernel_matrix(X, X)
        k_yy = self.compute_kernel_matrix(Y, Y)
        k_xy = self.compute_kernel_matrix(X, Y)
        
        # MMD^2 = E[K(x,x')] - 2*E[K(x,y)] + E[K(y,y')]
        mmd_squared = k_xx - 2 * k_xy + k_yy
        
        # Ensure non-negative (numerical stability)
        mmd_squared = max(0.0, mmd_squared)
        
        return np.sqrt(mmd_squared)
    
    def compute_mmd_vectorized(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Vectorized version of MMD computation for better performance
        
        Args:
            X: Samples from first distribution (n_samples_x, n_features)
            Y: Samples from second distribution (n_samples_y, n_features)
        
        Returns:
            MMD value
        """
        # Ensure inputs are 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Compute pairwise squared distances
        def pairwise_distances(A, B):
            """Compute pairwise squared Euclidean distances"""
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            A_sq = np.sum(A ** 2, axis=1, keepdims=True)
            B_sq = np.sum(B ** 2, axis=1, keepdims=True).T
            AB = A @ B.T
            return A_sq + B_sq - 2 * AB
        
        # Compute kernel matrices
        XX_dist = pairwise_distances(X, X)
        YY_dist = pairwise_distances(Y, Y)
        XY_dist = pairwise_distances(X, Y)
        
        # Apply Gaussian kernel
        K_XX = np.exp(-XX_dist / (2 * self.sigma ** 2))
        K_YY = np.exp(-YY_dist / (2 * self.sigma ** 2))
        K_XY = np.exp(-XY_dist / (2 * self.sigma ** 2))
        
        # Compute MMD
        k_xx = np.mean(K_XX)
        k_yy = np.mean(K_YY)
        k_xy = np.mean(K_XY)
        
        mmd_squared = k_xx - 2 * k_xy + k_yy
        mmd_squared = max(0.0, mmd_squared)
        
        return np.sqrt(mmd_squared)
    
    def __call__(self, X: np.ndarray, Y: np.ndarray, vectorized: bool = True) -> float:
        """
        Compute MMD loss
        
        Args:
            X: Samples from first distribution
            Y: Samples from second distribution
            vectorized: Use vectorized computation (faster)
        
        Returns:
            MMD value
        """
        if vectorized:
            return self.compute_mmd_vectorized(X, Y)
        else:
            return self.compute_mmd(X, Y)
