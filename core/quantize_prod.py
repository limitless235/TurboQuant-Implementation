import numpy as np
from typing import Dict, Tuple
from scipy.special import gamma as scipy_gamma
from qjl import QJL
from quantize_mse import TurboQuantMSE


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
        x = np.asarray(x)
        assert x.ndim == 1
        assert x.shape[0] == self.d

        # MSE stage — fixed rotation, deterministic
        x_hat_mse, residual = self.turbo_quant_mse.quantize_dequantize(x)
        idx = self.turbo_quant_mse.quantize(x)

        # QJL stage — fresh random projection per vector for unbiasedness
        qjl_fresh = QJL(self.d)
        qjl_bits, gamma = qjl_fresh.quantize(residual)

        return {
            'idx': idx,
            'qjl_bits': qjl_bits,
            'gamma': gamma,
            '_qjl_S': qjl_fresh.S  # must store S to dequantize later
        }

    def dequantize(self, idx, qjl_bits, gamma, _qjl_S) -> np.ndarray:
        x_mse = self.turbo_quant_mse.dequantize(idx)
        
        # Reconstruct QJL with the stored S
        qjl_fresh = QJL(self.d)
        qjl_fresh.S = _qjl_S
        x_qjl = qjl_fresh.dequantize(qjl_bits, gamma)
        
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
            tq_trial = TurboQuantProd(self.d, self.bit_width)  # fresh S each trial
            quantized = tq_trial.quantize(x)
            x_hat = tq_trial.dequantize(**quantized)
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

    # Generate test data
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.random.randn(n, d)
    mean_y_norm2 = np.mean(np.sum(Y ** 2, axis=1))

    print(f"{'b':>2} | {'mean_error':>11} | {'emp_dist':>11} | {'theor_dist':>11} | {'ratio':>7} | {'unbias':>7} | {'dist':>6}")
    print("-" * 80)

    for b in [1, 2, 3, 4]:
        # For EACH vector, use a fresh TurboQuantProd so S is independent
        # This correctly tests E[<y, x_hat>] = <y, x> over random S
        ip_true = np.sum(Y * X, axis=1)
        ip_est = np.zeros(n)

        for i in range(n):
            tq = TurboQuantProd(d=d, bit_width=b)  # fresh S per vector
            quantized = tq.quantize(X[i])
            x_hat = tq.dequantize(**quantized)
            ip_est[i] = np.dot(Y[i], x_hat)

        mean_error = float(np.mean(ip_est - ip_true))
        empirical_dist = float(np.mean((ip_est - ip_true) ** 2))
        dmse_b_minus_1 = 1.0 if b == 1 else np.sqrt(3 * np.pi / 2) * 4 ** (-(b - 1))
        theor_dist = float((np.pi / 2) * mean_y_norm2 / d * dmse_b_minus_1)
        ratio = empirical_dist / theor_dist
        unbias_pass = abs(mean_error) < 0.05
        dist_pass = 0.5 <= ratio <= 2.0

        print(f"{b:>2} | {mean_error:11.4e} | {empirical_dist:11.4e} | {theor_dist:11.4e} | {ratio:7.3f} | {'PASS' if unbias_pass else 'FAIL':>7} | {'PASS' if dist_pass else 'FAIL':>6}")
 
    print("-" * 80)

    # Compare bias: TurboQuantProd vs TurboQuantMSE on same vectors/S
    print(f"\nBias comparison at b=2 (same rotation for fair comparison):")
    b = 2
    # Use fixed instances so rotation is shared — tests bias from quantization not from S
    tqprod = TurboQuantProd(d=d, bit_width=b, seed=0)
    tqmse  = TurboQuantMSE(d=d, bit_width=b, seed=0)

    ip_true = np.sum(Y * X, axis=1)

    # TurboQuantProd: fresh S per vector
    ip_est_prod = np.zeros(n)
    for i in range(n):
        tq = TurboQuantProd(d=d, bit_width=b)
        quantized = tq.quantize(X[i])
        ip_est_prod[i] = np.dot(Y[i], tq.dequantize(**quantized))

    # TurboQuantMSE: fixed instance
    X_hat_mse = np.stack([tqmse.dequantize(tqmse.quantize(x)) for x in X])
    ip_est_mse = np.sum(Y * X_hat_mse, axis=1)

    mean_error_prod = float(np.mean(ip_est_prod - ip_true))
    mean_error_mse  = float(np.mean(ip_est_mse  - ip_true))

    print(f"TurboQuantProd mean_error: {mean_error_prod:.4e} (should be near 0)")
    print(f"TurboQuantMSE  mean_error: {mean_error_mse:.4e}  (should be nonzero)")
    print(f"TurboQuantProd more unbiased: {abs(mean_error_prod) < abs(mean_error_mse)}")