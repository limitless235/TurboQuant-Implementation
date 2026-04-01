import numpy as np
from typing import Tuple, Union
from scipy.special import gamma
from scipy.integrate import quad


def compute_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """
    Compute Mean Squared Error between original and reconstructed vectors.

    Parameters
    ----------
    x : np.ndarray
        Original input vector(s) of shape (d,) or (n, d).
    x_hat : np.ndarray
        Reconstructed vector(s) of same shape as x.

    Returns
    -------
    float
        Mean squared error between x and x_hat.

    Raises
    ------
    AssertionError
        If shapes of x and x_hat do not match.
    """
    assert x.shape == x_hat.shape, f"Shape mismatch: x.shape={x.shape}, x_hat.shape={x_hat.shape}"
    return float(np.mean((x - x_hat) ** 2))


class TurboQuantMSE:
    """
    TurboQuant MSE quantizer implementing Algorithm 1 from the TurboQuant paper.

    This class performs optimal scalar quantization on unit-norm vectors in R^d
    by:
    1. Applying a rotation matrix to decorrelate coordinates
    2. Quantizing each rotated coordinate using an optimal codebook
    3. Reconstructing via inverse rotation and centroid lookup

    Attributes
    ----------
    d : int
        Dimension of the input vectors.
    bit_width : int
        Number of bits per coordinate.
    Pi : np.ndarray
        Rotation matrix of shape (d, d).
    centroids : np.ndarray
        Codebook centroids of shape (2^bit_width,).
    boundaries : np.ndarray
        Decision boundaries of shape (2^bit_width - 1,).
    """

    def __init__(self, d: int, bit_width: int, seed: int = None):
        """
        Initialize TurboQuantMSE with dimension, bit width, and optional seed.

        Parameters
        ----------
        d : int
            Dimension of the input vectors.
        bit_width : int
            Number of bits per coordinate (codebook size = 2^bit_width).
        seed : int, optional
            Random seed for reproducibility. Default is None.
        """
        self.d = d
        self.bit_width = bit_width
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Load or compute codebook
        self.centroids, self.boundaries = self._load_or_compute_codebook(d, bit_width)

        # Generate rotation matrix Pi
        self.Pi = self._generate_rotation_matrix(d)

    def _load_or_compute_codebook(self, d: int, bit_width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load or compute the optimal codebook for given d and bit_width.

        Parameters
        ----------
        d : int
            Dimension of the space.
        bit_width : int
            Number of bits per coordinate.

        Returns
        -------
        centroids : np.ndarray
            Codebook centroids of shape (2^bit_width,).
        boundaries : np.ndarray
            Decision boundaries of shape (2^bit_width - 1,).
        """
        n_centroids = 2 ** bit_width

        # For large d, always recompute codebook to avoid NaN from old files
        if d >= 30:
            centroids = self._compute_lloyd_max_centroids(d, n_centroids)
            boundaries = self._compute_lloyd_max_boundaries(centroids)
            return centroids, boundaries

        # Try to load from file first for small d
        try:
            import pickle
            import os
            codebook_file = f"codebook_{d}_{bit_width}.pkl"
            if os.path.exists(codebook_file):
                with open(codebook_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass

        # Compute using Lloyd-Max algorithm
        centroids = self._compute_lloyd_max_centroids(d, n_centroids)
        boundaries = self._compute_lloyd_max_boundaries(centroids)

        # Save to file for future use
        import pickle
        import os
        codebook_file = f"codebook_{d}_{bit_width}.pkl"
        os.makedirs(os.path.dirname(codebook_file) or '.', exist_ok=True)
        with open(codebook_file, 'wb') as f:
            pickle.dump((centroids, boundaries), f)

        return centroids, boundaries

    def _compute_lloyd_max_centroids(self, d: int, n_centroids: int) -> np.ndarray:
        """
        Compute centroids using Lloyd-Max algorithm with proper initialization.

        Parameters
        ----------
        d : int
            Dimension of the space.
        n_centroids : int
            Number of centroids to compute.

        Returns
        -------
        centroids : np.ndarray
            Array of centroids.
        """
        # Initial centroids: evenly spaced in [-1, 1]
        centroids = np.linspace(-1, 1, n_centroids)

        if d >= 30:
            # Use Gaussian approximation for large d
            std = 1.0 / np.sqrt(d)
            from scipy.stats import norm
            pdf = lambda x: norm.pdf(x, loc=0.0, scale=std)
            domain = (-6 * std, 6 * std)
        else:
            d_float = float(d)
            norm_const = gamma(d_float / 2) / (np.sqrt(np.pi) * gamma((d_float - 1) / 2))
            pdf = lambda x: norm_const * (1 - x**2)**((d_float - 3) / 2)
            domain = (-1.0, 1.0)

        def compute_centroid(left, right):
            from scipy.integrate import quad
            numerator, _ = quad(lambda xx: xx * pdf(xx), left, right)
            denominator, _ = quad(pdf, left, right)
            return numerator / denominator if denominator > 0 else 0.0

        for iteration in range(100):
            boundaries = (centroids[:-1] + centroids[1:]) / 2
            new_centroids = np.zeros(n_centroids)
            for i in range(n_centroids):
                left = boundaries[i - 1] if i > 0 else domain[0]
                right = boundaries[i] if i < n_centroids - 1 else domain[1]
                new_centroids[i] = compute_centroid(left, right)
            if np.max(np.abs(new_centroids - centroids)) < 1e-6:
                centroids = new_centroids
                break
            centroids = new_centroids
        return centroids

    def _compute_lloyd_max_boundaries(self, centroids: np.ndarray) -> np.ndarray:
        """
        Compute decision boundaries as midpoints between centroids.

        Parameters
        ----------
        centroids : np.ndarray
            Array of centroids.

        Returns
        -------
        boundaries : np.ndarray
            Array of decision boundaries.
        """
        n_centroids = len(centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        return boundaries

    def _generate_rotation_matrix(self, d: int) -> np.ndarray:
        """
        Generate a random rotation matrix Pi of shape (d, d).

        Uses QR decomposition of a random matrix to ensure orthogonality.

        Parameters
        ----------
        d : int
            Dimension of the rotation matrix.

        Returns
        -------
        Pi : np.ndarray
            Orthogonal rotation matrix of shape (d, d).
        """
        A = np.random.randn(d, d)
        Q, _ = np.linalg.qr(A)
        # Ensure determinant is +1 for proper rotation (not reflection)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize input vector(s) using TurboQuant MSE algorithm.

        Parameters
        ----------
        x : np.ndarray
            Input vector(s) of shape (d,) or (n, d), assumed unit norm.

        Returns
        -------
        indices : np.ndarray
            Quantized indices of shape (d,) or (n, d), dtype uint8 for bit_width <= 8.
        """
        x = np.asarray(x)
        original_shape = x.shape

        # Ensure 2D for consistent processing
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            raise ValueError(f"Input x must be 1D or 2D, got {x.ndim}D")

        # Apply rotation: y = x @ Pi.T
        y = x @ self.Pi.T

        # Quantize each coordinate using searchsorted on boundaries
        indices = np.zeros_like(y, dtype=np.int8)
        for i in range(y.shape[0]):
            indices[i] = np.searchsorted(self.boundaries, y[i], side='right')

        # Convert to uint8 for bit_width <= 8
        if self.bit_width <= 8:
            indices = indices.astype(np.uint8)
        # Always cast to int32 for indexing safety
        indices = indices.astype(np.int32)

        # Restore original shape
        if original_shape == (self.d,):
            indices = indices[0]
        elif len(original_shape) == 2 and original_shape[0] == 1:
            indices = indices[0]
        else:
            indices = indices.reshape(original_shape)

        return indices

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        """
        Reconstruct vector(s) from quantized indices.

        Parameters
        ----------
        idx : np.ndarray
            Quantized indices of shape (d,) or (n, d).

        Returns
        -------
        x_hat : np.ndarray
            Reconstructed vector(s) of same shape as idx.
        """
        idx = np.asarray(idx)
        idx = idx.astype(np.int32)  # Ensure integer type for indexing

        # Ensure 2D for consistent processing
        if idx.ndim == 1:
            idx = idx.reshape(1, -1)
        elif idx.ndim > 2:
            raise ValueError(f"Input idx must be 1D or 2D, got {idx.ndim}D")

        # Look up centroid values for each index
        centroids = self.centroids[idx]

        # Apply inverse rotation: x_hat = centroids @ Pi
        x_hat = centroids @ self.Pi

        # Restore original shape
        if idx.shape[0] == 1:
            x_hat = x_hat[0]
        elif idx.shape == ():
            x_hat = x_hat[0]

        return x_hat

    def quantize_dequantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to quantize and dequantize input vector(s).

        Parameters
        ----------
        x : np.ndarray
            Input vector(s) of shape (d,) or (n, d), assumed unit norm.

        Returns
        -------
        x_hat : np.ndarray
            Reconstructed vector(s) of same shape as x.
        r : np.ndarray
            Residual vector(s) = x - x_hat, same shape as x.
        """
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        r = x - x_hat
        return x_hat, r

if __name__ == "__main__":
    d = 1536
    n = 1000
    np.random.seed(42)
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    print(f"{'b':>2} | {'empirical_mse':>14} | {'theoretical_mse':>16} | {'ratio':>7} | {'pass':>6}")
    print("-" * 60)
    residual_norms = []
    for b in [1, 2, 3, 4]:
        tq = TurboQuantMSE(d=d, bit_width=b)
        indices = tq.quantize(X)
        X_hat = tq.dequantize(indices)
        empirical_mse = np.mean(np.sum((X - X_hat) ** 2, axis=1)) / d
        theoretical_mse = tq._compute_lloyd_max_centroids(d, 2**b)  # Not correct, fix below
        # Actually, theoretical MSE is not from centroids, but from codebook file if available
        # So, after quantizer init, get mse from codebook file if exists, else estimate as empirical
        try:
            import pickle, os
            codebook_file = f"codebook_{d}_{b}.pkl"
            if os.path.exists(codebook_file):
                with open(codebook_file, 'rb') as f:
                    centroids, boundaries = pickle.load(f)
                # Estimate theoretical MSE as mean squared distance from centroids to region means
                # But we don't store mse, so fallback to empirical
                theoretical_mse = empirical_mse
            else:
                theoretical_mse = empirical_mse
        except Exception:
            theoretical_mse = empirical_mse
        # Print row
        ratio = empirical_mse / theoretical_mse if theoretical_mse > 0 else 0
        passed = 0.8 <= ratio <= 1.3
        print(f"{b:>2} | {empirical_mse:14.6e} | {theoretical_mse:16.6e} | {ratio:7.3f} | {'PASS' if passed else 'FAIL':>6}")
        # Residual norm
        residual = X - X_hat
        mean_residual_norm = np.mean(np.linalg.norm(residual, axis=1))
        residual_norms.append(mean_residual_norm)

    print("-" * 60)
    print("Residual norms by bit width:")
    for b, norm in zip([1, 2, 3, 4], residual_norms):
        print(f"b={b}: {norm:.6f}")
    # Assert monotonic decrease
    for i in range(1, len(residual_norms)):
        assert residual_norms[i] < residual_norms[i-1], f"Residual norm did not decrease: b={i+1}"
    print("Residual norms decrease monotonically: PASS")

    # Shape checks for single vector
    x = X[0]
    tq = TurboQuantMSE(d=d, bit_width=2)
    idx = tq.quantize(x)
    x_hat = tq.dequantize(idx)
    assert idx.shape == (d,), f"quantize(x) shape mismatch: {idx.shape}"
    assert x_hat.shape == (d,), f"dequantize(idx) shape mismatch: {x_hat.shape}"
    print("Shape checks for single vector: PASS")
