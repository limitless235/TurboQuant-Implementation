import numpy as np
from typing import Dict, Tuple
from scipy.special import gamma as scipy_gamma


class TurboQuantProd:
    """
    TurboQuant Product quantizer implementing Algorithm 2 from the TurboQuant paper.

    This class combines:
    1. MSE quantization (TurboQuantMSE) for the main signal
    2. QJL quantization for the residual

    The product scheme achieves better inner product preservation than MSE alone.

    Attributes
    ----------
    d : int
        Dimension of the input vectors.
    bit_width : int
        Total bits per coordinate.
    turbo_quant_mse : TurboQuantMSE
        MSE quantizer with bit_width - 1 bits.
    qjl : QJL
        QJL quantizer with same dimension d.
    """

    def __init__(self, d: int, bit_width: int, seed: int = None):
        """
        Initialize TurboQuantProd with dimension, bit width, and optional seed.

        Parameters
        ----------
        d : int
            Dimension of the input vectors.
        bit_width : int
            Total bits per coordinate.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        """
        self.d = d
        self.bit_width = bit_width
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Instantiate TurboQuantMSE with bit_width - 1
        self.turbo_quant_mse = TurboQuantMSE(d, bit_width - 1, seed=seed)

        # Instantiate QJL with same d
        self.qjl = QJL(d, seed=seed)

    def quantize(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Quantize input vector using TurboQuant product scheme.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (d,), assumed unit norm.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'idx': MSE quantized indices (d,)
            - 'qjl_bits': QJL sign bits (d,)
            - 'gamma': QJL norm scalar
        """
        x = np.asarray(x)
        assert x.ndim == 1, f"Input x must be 1D, got {x.ndim}D"
        assert x.shape[0] == self.d, f"Input x has wrong dimension: {x.shape[0]} vs {self.d}"

        # Run MSE quantizer
        idx, residual = self.turbo_quant_mse.quantize_dequantize(x)

        # Run QJL on residual
        qjl_bits, gamma = self.qjl.quantize(residual)

        return {
            'idx': idx,
            'qjl_bits': qjl_bits,
            'gamma': gamma
        }

    def dequantize(self, idx: np.ndarray, qjl_bits: np.ndarray, gamma: float) -> np.ndarray:
        """
        Reconstruct vector from quantized components.

        Parameters
        ----------
        idx : np.ndarray
            MSE quantized indices of shape (d,).
        qjl_bits : np.ndarray
            QJL sign bits of shape (d,).
        gamma : float
            QJL norm scalar.

        Returns
        -------
        x_hat : np.ndarray
            Reconstructed vector of shape (d,).
        """
        idx = np.asarray(idx)
        qjl_bits = np.asarray(qjl_bits)

        assert idx.shape == (self.d,), f"idx shape wrong: {idx.shape}"
        assert qjl_bits.shape == (self.d,), f"qjl_bits shape wrong: {qjl_bits.shape}"
        assert np.all(np.abs(qjl_bits) == 1), "qjl_bits must be {-1, +1}"

        # Reconstruct x_mse from idx
        x_mse = self.turbo_quant_mse.dequantize(idx)

        # Reconstruct x_qjl from (qjl_bits, gamma)
        x_qjl = self.qjl.dequantize(qjl_bits, gamma)

        # Return sum
        return x_mse + x_qjl

    def effective_bits(self) -> float:
        """
        Compute effective bits per coordinate.

        The product scheme uses (bit_width - 1) * d + d bits total,
        divided by d = bit_width.

        Returns
        -------
        float
            Effective bits per coordinate, equals self.bit_width.
        """
        return float(self.bit_width)

    def verify_unbiasedness(self, n_trials: int = 10000) -> Tuple[float, float, bool]:
        """
        Verify unbiasedness of the TurboQuantProd transform.

        For random unit vectors x and y, we verify that
        E[<y, QJL^-1(QJL(x))>] ≈ <y, x> within 3 standard errors.

        Parameters
        ----------
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
        # Generate random unit vectors
        x_raw = np.random.randn(self.d)
        x = x_raw / np.linalg.norm(x_raw)

        y_raw = np.random.randn(self.d)
        y = y_raw / np.linalg.norm(y_raw)

        # True inner product
        true_inner_product = float(np.dot(x, y))

        # Collect inner products over trials
        inner_products = []
        for _ in range(n_trials):
            # Quantize x
            quantized = self.quantize(x)

            # Dequantize
            x_hat = self.dequantize(quantized['idx'], quantized['qjl_bits'], quantized['gamma'])

            # Compute inner product with y
            ip = float(np.dot(y, x_hat))
            inner_products.append(ip)

        inner_products = np.array(inner_products)

        # Estimate bias
        estimated_bias = float(np.mean(inner_products) - true_inner_product)

        # Standard error
        std_error = np.std(inner_products) / np.sqrt(n_trials)

        # Check if bias is within 3 standard errors
        passed = abs(estimated_bias) <= 3 * std_error

        return estimated_bias, true_inner_product, passed

    def compute_inner_product_distortion(self, x: np.ndarray, y: np.ndarray, n_trials: int = 1000) -> Dict[str, float]:
        """
        Compute inner product distortion statistics.

        Parameters
        ----------
        x : np.ndarray
            First input vector of shape (d,).
        y : np.ndarray
            Second input vector of shape (d,).
        n_trials : int, optional
            Number of trials. Default is 1000.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'mean_distortion': Mean difference between true and estimated inner products
            - 'std_distortion': Standard deviation of distortion
            - 'max_distortion': Maximum absolute distortion
            - 'relative_error': Mean relative error
        """
        x = np.asarray(x)
        y = np.asarray(y)

        assert x.shape == (self.d,), f"x shape wrong: {x.shape}"
        assert y.shape == (self.d,), f"y shape wrong: {y.shape}"

        # True inner product
        true_ip = float(np.dot(x, y))

        # Collect distortions over trials
        distortions = []
        for _ in range(n_trials):
            # Quantize x
            quantized = self.quantize(x)

            # Dequantize
            x_hat = self.dequantize(quantized['idx'], quantized['qjl_bits'], quantized['gamma'])

            # Compute inner product with y
            estimated_ip = float(np.dot(y, x_hat))

            # Compute distortion
            distortion = estimated_ip - true_ip
            distortions.append(distortion)

        distortions = np.array(distortions)

        # Compute statistics
        mean_distortion = float(np.mean(distortions))
        std_distortion = float(np.std(distortions))
        max_distortion = float(np.max(np.abs(distortions)))
        relative_error = float(np.mean(np.abs(distortions) / np.abs(true_ip)))

        return {
            'mean_distortion': mean_distortion,
            'std_distortion': std_distortion,
            'max_distortion': max_distortion,
            'relative_error': relative_error
        }


if __name__ == "__main__":
    from quantize_mse import TurboQuantMSE
    d = 1536
    n = 1000
    np.random.seed(42)
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.random.randn(n, d)  # Not normalized
    mean_y_norm2 = np.mean(np.sum(Y ** 2, axis=1))
    prod_mean_errors = []
    mse_mean_errors = []
    print(f"{'b':>2} | {'mean_error':>11} | {'emp_dist':>11} | {'theor_dist':>11} | {'ratio':>7} | {'unbias':>7} | {'dist':>6}")
    print("-" * 80)
    for b in [1, 2, 3, 4]:
        tqprod = TurboQuantProd(d=d, bit_width=b)
        # Quantize and dequantize all x
        X_hat = np.stack([tqprod.dequantize(**tqprod.quantize(x)) for x in X])
        ip_true = np.sum(Y * X, axis=1)
        ip_est = np.sum(Y * X_hat, axis=1)
        mean_error = float(np.mean(ip_est - ip_true))
        empirical_dist = float(np.mean((ip_est - ip_true) ** 2))
        theor_dist = float(np.sqrt(3 * np.pi / 2) * mean_y_norm2 / d * 4 ** (-b))
        ratio = empirical_dist / theor_dist if theor_dist > 0 else 0
        unbias_pass = abs(mean_error) < 0.01
        dist_pass = 0.5 <= ratio <= 2.0
        prod_mean_errors.append(abs(mean_error))
        print(f"{b:>2} | {mean_error:11.4e} | {empirical_dist:11.4e} | {theor_dist:11.4e} | {ratio:7.3f} | {'PASS' if unbias_pass else 'FAIL':>7} | {'PASS' if dist_pass else 'FAIL':>6}")
    print("-" * 80)
    # Compare with TurboQuantMSE for b=2
    b = 2
    tqprod = TurboQuantProd(d=d, bit_width=b)
    tqmse = TurboQuantMSE(d=d, bit_width=b)
    X_hat_prod = np.stack([tqprod.dequantize(**tqprod.quantize(x)) for x in X])
    X_hat_mse = np.stack([tqmse.dequantize(tqmse.quantize(x)) for x in X])
    ip_true = np.sum(Y * X, axis=1)
    ip_est_prod = np.sum(Y * X_hat_prod, axis=1)
    ip_est_mse = np.sum(Y * X_hat_mse, axis=1)
    mean_error_prod = float(np.mean(ip_est_prod - ip_true))
    mean_error_mse = float(np.mean(ip_est_mse - ip_true))
    print(f"\nComparison for b=2:")
    print(f"TurboQuantProd mean_error: {mean_error_prod:.4e} (should be near 0, UNBIASED)")
    print(f"TurboQuantMSE  mean_error: {mean_error_mse:.4e} (should be nonzero, BIASED)")
    mse_mean_errors.append(abs(mean_error_mse))
    # Assert unbiasedness
    assert abs(mean_error_prod) < abs(mean_error_mse), "TurboQuantProd should be more unbiased than TurboQuantMSE!"
    # Also check for all bit widths
    for b in [1, 2, 3, 4]:
        tqprod = TurboQuantProd(d=d, bit_width=b)
        tqmse = TurboQuantMSE(d=d, bit_width=b)
        X_hat_prod = np.stack([tqprod.dequantize(**tqprod.quantize(x)) for x in X])
        X_hat_mse = np.stack([tqmse.dequantize(tqmse.quantize(x)) for x in X])
        ip_true = np.sum(Y * X, axis=1)
        ip_est_prod = np.sum(Y * X_hat_prod, axis=1)
        ip_est_mse = np.sum(Y * X_hat_mse, axis=1)
        mean_error_prod = float(np.mean(ip_est_prod - ip_true))
        mean_error_mse = float(np.mean(ip_est_mse - ip_true))
        prod_mean_errors.append(abs(mean_error_prod))
        mse_mean_errors.append(abs(mean_error_mse))
        assert abs(mean_error_prod) < abs(mean_error_mse), f"TurboQuantProd should be more unbiased than TurboQuantMSE for b={b}!"
    print("\nTurboQuantProd is more unbiased than TurboQuantMSE for all bit widths: PASS")

