import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from quantize_mse import TurboQuantMSE
from quantize_prod import TurboQuantProd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_benchmark():
    """Run benchmarking script for TurboQuant distortion analysis."""
    
    # Generate synthetic data
    d = 1536
    n = 200
    np.random.seed(42)
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.random.randn(n, d)
    mean_y_norm2 = float(np.mean(np.sum(Y**2, axis=1)))
    
    # Fine-grained theoretical values from paper Section 1.3
    fine_grained = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}
    
    # Results storage
    results = {
        'bit_width': [],
        'emp_mse': [],
        'ub_mse': [],
        'lb_mse': [],
        'emp_ip_mse': [],
        'emp_ip_prod': [],
        'ub_ip': [],
        'lb_ip': [],
        'mse_ok': [],
        'ip_ok': []
    }
    
    print("=" * 120)
    print("TURBOQUANT DISTORTION BENCHMARK")
    print("=" * 120)
    
    # For each bit_width
    for b in [1, 2, 3, 4, 5]:
        print(f"\n{'='*120}")
        print(f"Processing bit_width = {b}")
        print(f"{'='*120}")
        
        # 1. Create fixed TurboQuantMSE instance
        tqmse = TurboQuantMSE(d=d, bit_width=b, seed=0)
        
        # 2. Compute empirical MSE
        x_hats_mse = []
        for x in X:
            x_hat, _ = tqmse.quantize_dequantize(x)
            x_hats_mse.append(x_hat)
        x_hats_mse = np.array(x_hats_mse)
        emp_mse = float(np.mean(np.sum((X - x_hats_mse)**2, axis=1)))
        results['emp_mse'].append(emp_mse)
        
        # 3. Compute empirical inner product distortion for MSE
        ip_distortions_mse = []
        for i in range(n):
            x = X[i]
            y = Y[i]
            x_hat, _ = tqmse.quantize_dequantize(x)
            true_ip = float(np.dot(y, x))
            estimated_ip = float(np.dot(y, x_hat))
            distortion = true_ip - estimated_ip
            ip_distortions_mse.append(distortion**2)
        emp_ip_mse = float(np.mean(ip_distortions_mse))
        results['emp_ip_mse'].append(emp_ip_mse)
        
        # 4. Compute empirical inner product distortion for Prod (fresh instance per vector)
        ip_distortions_prod = []
        for i in range(n):
            x = X[i]
            y = Y[i]
            tq = TurboQuantProd(d=d, bit_width=b)
            quantized = tq.quantize(x)
            x_hat = tq.dequantize(**quantized)
            true_ip = float(np.dot(y, x))
            estimated_ip = float(np.dot(y, x_hat))
            distortion = true_ip - estimated_ip
            ip_distortions_prod.append(distortion**2)
        emp_ip_prod = float(np.mean(ip_distortions_prod))
        results['emp_ip_prod'].append(emp_ip_prod)
        
        # 5. Compute theoretical bounds
        ub_mse = np.sqrt(3 * np.pi / 2) * (4 ** (-b))
        lb_mse = 4 ** (-b)
        results['ub_mse'].append(ub_mse)
        results['lb_mse'].append(lb_mse)
        
        # Fine-grained inner product bounds
        if b in fine_grained:
            ub_ip_base = fine_grained[b]
        else:
            ub_ip_base = np.sqrt(3 * np.pi / 2) * (4 ** (-(b - 1)))
        ub_ip = ub_ip_base * mean_y_norm2 / d
        lb_ip = mean_y_norm2 / d * (4 ** (-b))
        results['ub_ip'].append(ub_ip)
        results['lb_ip'].append(lb_ip)
        
        # Check MSE bounds
        mse_ok = lb_mse * 0.5 <= emp_mse <= ub_mse * 3
        results['mse_ok'].append(mse_ok)
        
        # Check IP bounds
        ip_ok = lb_ip * 0.5 <= emp_ip_prod <= ub_ip * 3
        results['ip_ok'].append(ip_ok)
        results['bit_width'].append(b)
        
        # Print results
        print(f"\n  Bit Width: {b}")
        print(f"  Empirical MSE: {emp_mse:.6f}")
        print(f"  Theoretical MSE Upper: {ub_mse:.6f}")
        print(f"  Theoretical MSE Lower: {lb_mse:.6f}")
        print(f"  Empirical IP Distortion (MSE): {emp_ip_mse:.6f}")
        print(f"  Empirical IP Distortion (Prod): {emp_ip_prod:.6f}")
        print(f"  Theoretical IP Upper: {ub_ip:.6f}")
        print(f"  Theoretical IP Lower: {lb_ip:.6f}")
        print(f"  MSE OK: {'PASS' if mse_ok else 'FAIL'}")
        print(f"  IP OK: {'PASS' if ip_ok else 'FAIL'}")
    
    # Print final table
    print("\n" + "=" * 120)
    print("FINAL RESULTS TABLE")
    print("=" * 120)
    print(f"{'b':<6} {'emp_mse':<12} {'ub_mse':<12} {'lb_mse':<12} {'emp_ip_prod':<14} {'ub_ip':<12} {'lb_ip':<12} {'mse_ok':<8} {'ip_ok':<8}")
    print("-" * 120)
    
    for i in range(len(results['bit_width'])):
        b = results['bit_width'][i]
        emp_mse = results['emp_mse'][i]
        ub_mse = results['ub_mse'][i]
        lb_mse = results['lb_mse'][i]
        emp_ip_prod = results['emp_ip_prod'][i]
        ub_ip = results['ub_ip'][i]
        lb_ip = results['lb_ip'][i]
        mse_ok = 'PASS' if results['mse_ok'][i] else 'FAIL'
        ip_ok = 'PASS' if results['ip_ok'][i] else 'FAIL'
        print(f"{b:<6} {emp_mse:<12.6f} {ub_mse:<12.6f} {lb_mse:<12.6f} {emp_ip_prod:<14.6f} {ub_ip:<12.6f} {lb_ip:<12.6f} {mse_ok:<8} {ip_ok:<8}")
    
    # Create figures directory
    os.makedirs('bench/figures', exist_ok=True)
    
    # Create plot matching Figure 3 of the paper
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Inner product error
    ax1 = axes[0]
    ax1.set_xlabel('Bit Width', fontsize=12)
    ax1.set_ylabel('Inner Product Distortion', fontsize=12)
    ax1.set_title('Inner Product Distortion', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', alpha=0.3)
    
    # Plot TurboQuantMSE empirical (blue solid)
    ax1.plot(results['bit_width'], results['emp_ip_mse'], 'o-', label='TurboQuantMSE', linewidth=2, color='blue')
    # Plot TurboQuantProd empirical (purple solid)
    ax1.plot(results['bit_width'], results['emp_ip_prod'], 's-', label='TurboQuantProd', linewidth=2, color='purple')
    # Plot upper bound (red dashed)
    ax1.plot(results['bit_width'], results['ub_ip'], 'd--', label='Upper Bound', linewidth=1, alpha=0.7, color='red')
    # Plot lower bound (green dashed)
    ax1.plot(results['bit_width'], results['lb_ip'], 'd--', label='Lower Bound', linewidth=1, alpha=0.7, color='green')
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlabel('Bit Width', fontsize=12)
    ax1.set_ylabel('Inner Product Distortion', fontsize=12)
    ax1.set_title('Inner Product Distortion', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', alpha=0.3)
    
    # Right subplot: MSE
    ax2 = axes[1]
    ax2.set_xlabel('Bit Width', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Mean Squared Error', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', alpha=0.3)
    
    # Plot TurboQuantMSE empirical (blue solid)
    ax2.plot(results['bit_width'], results['emp_mse'], 'o-', label='TurboQuantMSE', linewidth=2, color='blue')
    # Plot upper bound (red dashed)
    ax2.plot(results['bit_width'], results['ub_mse'], 'd--', label='Upper Bound', linewidth=1, alpha=0.7, color='red')
    # Plot lower bound (green dashed)
    ax2.plot(results['bit_width'], results['lb_mse'], 'd--', label='Lower Bound', linewidth=1, alpha=0.7, color='green')
    
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlabel('Bit Width', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Mean Squared Error', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bench/figures/distortion_validation.png', dpi=300, bbox_inches='tight')
    print("\nSaved figure to bench/figures/distortion_validation.png")
    
    return results


if __name__ == "__main__":
    results = run_benchmark()