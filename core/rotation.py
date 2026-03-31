import numpy as np
from typing import Tuple, Union


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
    return np.mean((x - x_hat) ** 2)


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
        try:
            from core.codebook import load_codebook
            centroids, boundaries = load_codebook(d, bit_width)
        except ImportError:
            # Fallback: generate using Lloyd-Max if core module not available
            from scipy.special import gamma
            from scipy.integrate import quad

            n_centroids = 2 ** bit_width
            centroids = np.linspace(-1, 1, n_centroids + 1)[1:-1]

            def beta_distribution_pdf(x, d):
                d = float(d)
                norm_const = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
                return norm_const * (1 - x**2)**((d - 3) / 2)

            def compute_centroid(left, right, d):
                pdf = lambda x: beta_distribution_pdf(x, d)
                numerator, _ = quad(lambda x: x * pdf(x), left, right)
                denominator, _ = quad(pdf, left, right)
                return numerator / denominator

            for _ in range(100):
                boundaries = np.zeros(n_centroids - 1)
                for i in range(n_centroids - 1):
                    boundaries[i] = (centroids[i] + centroids[i + 1]) / 2

                new_centroids = np.zeros(n_centroids)
                for i in range(n_centroids):
                    left = boundaries[i - 1] if i > 0 else -1
                    right = boundaries[i] if i < n_centroids - 1 else 1
                    new_centroids[i] = compute_centroid(left, right, d)

                centroids = new_centroids

            centroids, boundaries = centroids, boundaries

        return centroids, boundaries

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

        # Apply rotation: y = Pi @ x
        y = self.Pi @ x

        # Quantize each coordinate using searchsorted on boundaries
        indices = np.zeros_like(y, dtype=np.int8)
        for i in range(y.shape[0]):
            # Find which region each coordinate falls into
            indices[i] = np.searchsorted(self.boundaries, y[i], side='right')

        # Convert to uint8 for bit_width <= 8
        if self.bit_width <= 8:
            indices = indices.astype(np.uint8)

        # Restore original shape
        if original_shape == ():
            indices = indices[0]
        elif original_shape[0] == 1:
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

        # Ensure 2D for consistent processing
        if idx.ndim == 1:
            idx = idx.reshape(1, -1)
        elif idx.ndim > 2:
            raise ValueError(f"Input idx must be 1D or 2D, got {idx.ndim}D")

        # Look up centroid values for each index
        centroids = self.centroids[idx]

        # Apply inverse rotation: x_hat = Pi.T @ centroids
        x_hat = self.Pi.T @ centroids

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


# Example usage and validation
if __name__ == "__main__":
    # Initialize TurboQuantMSE
    d = 128
    bit_width = 4
    turbo = TurboQuantMSE(d, bit_width, seed=42)

    # Generate test vector
    x = np.random.randn(d)
    x = x / np.linalg.norm(x)  # Normalize to unit norm

    # Quantize and dequantize
    x_hat, residual = turbo.quantize_dequantize(x)

    # Compute MSE
    mse = compute_mse(x, x_hat)
    print(f"MSE: {mse:.6f}")

    # Verify shapes
    assert x.shape == x_hat.shape, "Shape mismatch!"
    assert x.shape == residual.shape, "Residual shape mismatch!"

    # Test with batch
    x_batch = np.random.randn(10, d)
    x_batch = x_batch / np.linalg.norm(x_batch, axis=1, keepdims=True)
    indices_batch = turbo.quantize(x_batch)
    x_hat_batch = turbo.dequantize(indices_batch)
    x_hat_batch, residual_batch = turbo.quantize_dequantize(x_batch)

    assert x_batch.shape == indices_batch.shape, "Batch shape mismatch!"
    assert x_batch.shape == x_hat_batch.shape, "Batch reconstructed shape mismatch!"
    assert x_batch.shape == residual_batch.shape, "Batch residual shape mismatch!"

    print("All tests passed!")