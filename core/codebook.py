import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad
from scipy.stats import norm as scipy_norm
import os
import warnings
from scipy.integrate import IntegrationWarning


# Suppress integration warnings after we've handled them properly
warnings.filterwarnings('ignore', category=IntegrationWarning)


def beta_distribution_pdf(x, d):
    """
    Compute the PDF of a single coordinate of a random unit vector in R^d.

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    for x in [-1, 1].

    For d >= 30, uses the Gaussian approximation N(0, 1/d) which is
    numerically equivalent but avoids overflow/singularity issues.

    Parameters
    ----------
    x : float
        Point at which to evaluate the PDF.
    d : int
        Dimension of the space.

    Returns
    -------
    float
        The PDF value at x.
    """
    if d >= 30:
        # Gaussian approximation: N(0, 1/d), i.e. std = 1/sqrt(d)
        std = 1.0 / np.sqrt(d)
        return scipy_norm.pdf(x, loc=0.0, scale=std)

    # Exact Beta formula using log-space arithmetic to avoid gamma overflow
    if abs(x) >= 1.0:
        return 0.0
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    log_val = log_norm + ((d - 3) / 2) * np.log(1.0 - x**2)
    return np.exp(log_val)


def _make_pdf(d):
    """Return a scalar pdf function and its integration domain."""
    if d >= 30:
        std = 1.0 / np.sqrt(d)
        # Truncate Gaussian at ±6 sigma for numerical safety
        domain = (-6 * std, 6 * std)
        return scipy_norm(loc=0.0, scale=std).pdf, domain
    else:
        return lambda x: beta_distribution_pdf(x, d), (-1.0, 1.0)


def _integrate(f, left, right, d):
    """
    Numerically integrate f over [left, right] with appropriate settings.

    For peaked distributions (large d), passes the peak location as a hint
    to quad so it doesn't miss the mass.
    """
    std = 1.0 / np.sqrt(d)
    # Hint: the PDF peaks at 0, pass it if it's inside the interval
    points = [p for p in [0.0] if left < p < right]
    val, _ = quad(f, left, right, points=points, limit=200,
                  epsabs=1e-12, epsrel=1e-10)
    return val


def compute_centroid(left, right, d):
    """
    Compute the centroid (mean) of a quantization region under the marginal PDF.

    Centroid = E[X | X in [left, right]] = integral(x * f(x)) / integral(f(x))

    Parameters
    ----------
    left : float
        Left boundary of the region.
    right : float
        Right boundary of the region.
    d : int
        Dimension of the space.

    Returns
    -------
    float
        The centroid of the region.
    """
    pdf, _ = _make_pdf(d)
    numerator = _integrate(lambda x: x * pdf(x), left, right, d)
    denominator = _integrate(pdf, left, right, d)
    if denominator < 1e-300:
        # Region has essentially zero mass — return midpoint as fallback
        return (left + right) / 2.0
    return numerator / denominator


def compute_region_mse(left, right, centroid, d):
    """
    Compute the MSE contribution of one quantization region.

    MSE_region = integral((x - centroid)^2 * f(x), left, right)
               = E[X^2] - 2*centroid*E[X] + centroid^2 * P(region)

    Note: this is NOT E[X^2] - centroid^2 * P(region), which is a common
    mistake. The correct expansion of E[(X-c)^2] is E[X^2] - 2c*E[X] + c^2.

    Parameters
    ----------
    left : float
        Left boundary.
    right : float
        Right boundary.
    centroid : float
        Centroid of this region.
    d : int
        Dimension.

    Returns
    -------
    float
        MSE contribution from this region.
    """
    pdf, _ = _make_pdf(d)
    # Direct integration of (x - centroid)^2 * f(x) — simplest and most correct
    mse_region = _integrate(lambda x: (x - centroid)**2 * pdf(x), left, right, d)
    return mse_region


def lloyd_max_quantizer(b, d, tol=1e-8, max_iter=200):
    """
    Generate an optimal scalar quantizer using the Lloyd-Max algorithm.

    Alternates between:
    1. Updating boundaries as midpoints between adjacent centroids.
    2. Updating centroids as conditional means within each region.

    Parameters
    ----------
    b : int
        Bit width.
    d : int
        Dimension of the space.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    centroids : ndarray, shape (2^b,)
        Sorted centroid values.
    boundaries : ndarray, shape (2^b - 1,)
        Interior decision boundaries (not including -1 and +1).
    mse : float
        Total MSE = sum of region MSE contributions = d * C(fX, b).
        To get C(fX, b) (the per-coordinate cost), divide by d.
    """
    k = 2 ** b  # number of centroids

    # --- Special case b=1: analytical solution ---
    if b == 1:
        if d >= 30:
            # Gaussian: optimal 1-bit centroids are ±sqrt(2/pi) * std
            std = 1.0 / np.sqrt(d)
            a = np.sqrt(2.0 / np.pi) * std
        else:
            # Numerical solution for small d
            a = compute_centroid(0.0, 1.0, d)
        centroids = np.array([-a, a])
        boundaries = np.array([0.0])
        mse = (compute_region_mse(-1.0, 0.0, centroids[0], d) +
               compute_region_mse(0.0, 1.0, centroids[1], d))
        return centroids, boundaries, mse

    # --- General case: Lloyd-Max iterations ---

    # Initialise centroids uniformly, avoiding the ±1 boundaries
    _, domain = _make_pdf(d)
    lo, hi = domain
    centroids = np.linspace(lo * 0.9, hi * 0.9, k)

    for iteration in range(max_iter):
        # Step 1: update interior boundaries as midpoints
        # boundaries has k-1 elements (interior only)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0  # shape (k-1,)

        # Full boundary array including outer edges for region integration
        full_bounds = np.concatenate([[-1.0], boundaries, [1.0]])  # shape (k+1,)

        # Step 2: update centroids as conditional means
        new_centroids = np.zeros(k)
        for i in range(k):
            left = full_bounds[i]
            right = full_bounds[i + 1]
            new_centroids[i] = compute_centroid(left, right, d)

        delta = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if delta < tol:
            break

    # Enforce exact symmetry (the distribution is symmetric around 0)
    # Average with negated-and-flipped version
    centroids = (centroids - centroids[::-1]) / 2.0

    # Recompute boundaries after symmetry enforcement
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    full_bounds = np.concatenate([[-1.0], boundaries, [1.0]])

    # Compute total MSE
    mse = 0.0
    for i in range(k):
        left = full_bounds[i]
        right = full_bounds[i + 1]
        mse += compute_region_mse(left, right, centroids[i], d)

    return centroids, boundaries, mse


def precompute_codebooks(save_path='codebooks.npz'):
    """
    Precompute codebooks for various bit widths and dimensions and save to disk.

    Keys in the .npz file are strings of the form 'b{b}_d{d}_centroids',
    'b{b}_d{d}_boundaries', 'b{b}_d{d}_mse'.

    Parameters
    ----------
    save_path : str
        Path to save the .npz file.

    Returns
    -------
    results : dict
        Dictionary keyed by (b, d) tuples with sub-dicts containing
        'centroids', 'boundaries', 'mse'.
    """
    b_values = [1, 2, 3, 4, 5]
    d_values = [200, 512, 1536, 3072]

    results = {}
    flat_results = {}  # string-keyed for np.savez

    for b in b_values:
        for d in d_values:
            print(f"  Computing b={b}, d={d}...", end=' ', flush=True)
            centroids, boundaries, mse = lloyd_max_quantizer(b, d)
            results[(b, d)] = {
                'centroids': centroids,
                'boundaries': boundaries,
                'mse': mse
            }
            # FIX: np.savez requires string keys — use a formatted string
            prefix = f'b{b}_d{d}'
            flat_results[f'{prefix}_centroids'] = centroids
            flat_results[f'{prefix}_boundaries'] = boundaries
            flat_results[f'{prefix}_mse'] = np.array([mse])
            print(f"MSE={mse:.6f}, MSE*d={mse*d:.4f}")

    if os.path.exists(save_path):
        os.remove(save_path)

    np.savez(save_path, **flat_results)
    print(f"\nCodebooks saved to {save_path}")
    return results


def load_codebooks(load_path='codebooks.npz'):
    """
    Load precomputed codebooks from disk.

    Parameters
    ----------
    load_path : str
        Path to the .npz file produced by precompute_codebooks().

    Returns
    -------
    dict keyed by (b, d) tuples.
    """
    data = np.load(load_path)
    results = {}
    # Parse keys of the form 'b{b}_d{d}_centroids' etc.
    keys_seen = set()
    for key in data.files:
        parts = key.split('_')
        b = int(parts[0][1:])
        d = int(parts[1][1:])
        keys_seen.add((b, d))

    for (b, d) in keys_seen:
        prefix = f'b{b}_d{d}'
        results[(b, d)] = {
            'centroids':  data[f'{prefix}_centroids'],
            'boundaries': data[f'{prefix}_boundaries'],
            'mse':        float(data[f'{prefix}_mse'][0])
        }
    return results


def validate_codebooks(results):
    """
    Validate precomputed codebooks against known theoretical values from the paper.

    Expected MSE * d values (from paper Section 1.3):
        b=1: ~0.36
        b=2: ~0.117
        b=3: ~0.03
        b=4: ~0.009

    Parameters
    ----------
    results : dict
        Dictionary keyed by (b, d).

    Returns
    -------
    bool
        True if all checks pass.
    """
    # Expected MSE * d from the paper (Table in Section 1.3)
    expected_mse_times_d = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    rtol = 0.12  # 12% tolerance — paper values are approximate

    print("\n--- Validation Table ---")
    print(f"{'b':>3} {'d':>6} {'MSE*d (got)':>14} {'MSE*d (expected)':>18} {'ratio':>8} {'pass':>6}")
    print("-" * 60)

    all_passed = True
    for b in [1, 2, 3, 4]:
        expected = expected_mse_times_d[b]
        for d in [200, 512, 1536, 3072]:
            key = (b, d)
            if key not in results:
                continue
            mse = results[key]['mse']
            mse_times_d = mse * d
            ratio = mse_times_d / expected
            passed = abs(ratio - 1.0) < rtol
            if not passed:
                all_passed = False
            print(f"{b:>3} {d:>6} {mse_times_d:>14.4f} {expected:>18.4f} {ratio:>8.3f} {'OK' if passed else 'FAIL':>6}")

            # Extra check for b=1: centroids should be ±sqrt(2/pi)/sqrt(d)
            if b == 1:
                centroids = results[key]['centroids']
                expected_pos = np.sqrt(2.0 / np.pi) / np.sqrt(d)
                c_ratio = centroids[1] / expected_pos
                if abs(c_ratio - 1.0) > 0.05:
                    print(f"  WARNING b=1 d={d}: centroid ratio={c_ratio:.4f} "
                          f"(expected 1.0, got {centroids[1]:.6f} vs {expected_pos:.6f})")

    print("-" * 60)
    print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("Precomputing TurboQuant codebooks...")
    results = precompute_codebooks()
    validate_codebooks(results)