import numpy as np
from typing import Tuple
from scipy.special import gamma as scipy_gamma


class QJL:
    """
    Quantized Johnson-Lindenstrauss (QJL) transform as described in Definition 1
    of the TurboQuant paper.

    The QJL transform provides a randomized dimensionality reduction with
    quantized representation:
    1. Random rotation via S (d×d Gaussian matrix)
    2. Sign-based quantization to {-1, +1}
    3. Inverse transform with proper scaling

    Attributes
    ----------
    d : int
        Dimension of the space.
    S : np.ndarray
        Random matrix of shape (d, d) with i.i.d. N(0,1) entries.
    """

    def __init__(self, d: int, seed: int = None):
        """
        Initialize QJL with dimension and optional seed.

        Parameters
        ----------
        d : int
            Dimension of the space.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        """
        self.d = d
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Generate random matrix S with i.i.d. N(0,1) entries
        self.S = np.random.randn(d, d)


    def quantize(self, r: np.ndarray) -> Tuple[np.ndarray, float]:
        r = np.asarray(r)
        assert r.ndim == 1
        assert r.shape[0] == self.d

        gamma = float(np.linalg.norm(r))
        
        # Normalize before rotation so sign bits encode direction only
        # gamma carries the magnitude separately
        if gamma > 1e-10:
            r_normalized = r / gamma
        else:
            r_normalized = r
        
        rotated = self.S @ r_normalized
        sign_bits = np.sign(rotated).astype(np.int8)
        
        return sign_bits, gamma

    def dequantize(self, sign_bits: np.ndarray, gamma: float) -> np.ndarray:
        """
        Dequantize sign bits back to approximate original vector.

        Parameters
        ----------
        sign_bits : np.ndarray
            Quantized signs of shape (d,), dtype int8 with values {-1, +1}.
        gamma : float
            Stored L2 norm of the original vector.

        Returns
        -------
        x_hat : np.ndarray
            Reconstructed vector of shape (d,), dtype float.
        """
        sign_bits = np.asarray(sign_bits)
        assert sign_bits.ndim == 1, f"Input sign_bits must be 1D, got {sign_bits.ndim}D"
        assert sign_bits.shape[0] == self.d, f"Input sign_bits has wrong dimension: {sign_bits.shape[0]} vs {self.d}"
        assert np.all(np.abs(sign_bits) == 1), "sign_bits must be {-1, +1}"
        zeros = np.sum(sign_bits == 0)
        if zeros > 0:
            print(f"WARNING: {zeros} zero sign bits")

        # Apply inverse transform: sqrt(pi/2) / d * gamma * S.T @ sign_bits
        x_hat = np.sqrt(np.pi / 2) / self.d * gamma * (self.S.T @ sign_bits)



        return x_hat


def verify_unbiasedness(d: int, n_trials: int = 10000) -> Tuple[float, float, bool]:
    """
    Verify unbiasedness of the QJL transform.

    For random unit vectors x and y, we verify that E[<y, QJL^-1(QJL(x))>] ≈ <y, x>
    within 3 standard errors.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n_trials : int, optional
        Number of trials for Monte Carlo estimation. Default is 10000.

    Returns
    -------
    estimated_bias : float
        Estimated bias (mean difference between inner products).
    true_inner_product : float
        True inner product <y, x>.
    passed : bool
        True if bias is within 3 standard errors of zero.
    """
    qjl = QJL(d)


    # Generate random unit vectors
    x_raw = np.random.randn(d)
    x = x_raw / np.linalg.norm(x_raw)

    y_raw = np.random.randn(d)
    y = y_raw / np.linalg.norm(y_raw)

    # True inner product
    true_inner_product = float(np.dot(x, y))

    # Collect inner products over trials
    inner_products = []
    for _ in range(n_trials):
        qjl_trial = QJL(d)  # fresh S each trial
        sign_bits, gamma = qjl_trial.quantize(x)
        x_hat = qjl_trial.dequantize(sign_bits, gamma)
        ip = float(np.dot(y, x_hat))
        inner_products.append(ip)

    inner_products = np.array(inner_products)

    # Estimate bias
    estimated_bias=float(np.mean(inner_products) - true_inner_product)


    # Standard error
    std_error = np.std(inner_products) / np.sqrt(n_trials)

    # Check if bias is within 3 standard errors
    passed = abs(estimated_bias) <= 3 * std_error
    # After computing inner_products:
    print(f"  mean(x_hat): {np.mean([np.dot(y, qjl.dequantize(*qjl.quantize(x))) for _ in range(100)]):.4f}")
    print(f"  true <y,x>:  {true_inner_product:.4f}")
    print(f"  std_error:   {std_error:.6f}")
    print(f"  |bias|/std:  {abs(estimated_bias)/std_error:.2f}")

    return estimated_bias, true_inner_product, passed


if __name__ == "__main__":
    # Test QJL with different dimensions
    for d in [64, 128, 256]:
        qjl = QJL(d, seed=42)

        # Test quantize/dequantize
        x = np.random.randn(d)
        x = x / np.linalg.norm(x)

        sign_bits, gamma = qjl.quantize(x)
        x_hat = qjl.dequantize(sign_bits, gamma)

        # Check shapes
        assert sign_bits.shape == (d,), f"sign_bits shape wrong: {sign_bits.shape}"
        assert x_hat.shape == (d,), f"x_hat shape wrong: {x_hat.shape}"
        assert np.all(np.abs(sign_bits) == 1), "sign_bits not {-1, +1}"

        # Verify norm preservation approximately
        print(f"d={d}: ||x||={np.linalg.norm(x):.6f}, ||x_hat||={np.linalg.norm(x_hat):.6f}")

        # Test verify_unbiasedness
        estimated_bias, true_ip, passed = verify_unbiasedness(d, n_trials=10000)
        print(f"d={d}: estimated_bias={estimated_bias:.6f}, true_ip={true_ip:.6f}, passed={passed}")

    print("\nAll tests completed!")
