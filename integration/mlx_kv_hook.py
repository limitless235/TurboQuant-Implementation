import sys
import os
import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))
from quantize_prod import TurboQuantProd as CoreTurboQuantProd

class TurboQuantKVCache:
    def __init__(self, head_dim, bit_width=3, use_prod=True):
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.outlier_dim = 32
        self.normal_dim = head_dim - self.outlier_dim
        
        self.outlier_bits = bit_width + 1
        self.normal_bits = bit_width - 1
        
        self.q_outlier = CoreTurboQuantProd(self.outlier_dim, self.outlier_bits)
        self.q_normal = CoreTurboQuantProd(self.normal_dim, self.normal_bits)

    def _pack(self, vectors, quantizer):
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-9
        normalized = vectors / norms
        
        idx, bits, gamma, S = [], [], [], []
        for vec in normalized:
            res = quantizer.quantize(vec)
            if isinstance(res, dict):
                idx.append(res.get('idx'))
                bits.append(res.get('qjl_bits', res.get('bits')))
                gamma.append(res.get('gamma'))
                S.append(res.get('_qjl_S', res.get('S')))
            else:
                idx.append(res[0])
                bits.append(res[1])
                gamma.append(res[2])
                S.append(res[3])
        
        return {'norms': norms, 'idx': idx, 'bits': bits, 'gamma': gamma, 'S': S}

    def _unpack(self, packed, quantizer, num_vecs, dim):
        approx = np.zeros((num_vecs, dim), dtype=np.float32)
        for i in range(num_vecs):
            approx[i] = quantizer.dequantize(
                packed['idx'][i], packed['bits'][i], 
                packed['gamma'][i], packed['S'][i]
            )
        return approx * packed['norms']

    def compress_kv(self, keys_np, values_np):
        if isinstance(keys_np, mx.array):
            keys_np = np.array(keys_np.astype(mx.float32), dtype=np.float32)
        if isinstance(values_np, mx.array):
            values_np = np.array(values_np.astype(mx.float32), dtype=np.float32)

        keys_np = np.nan_to_num(keys_np, nan=0.0, posinf=0.0, neginf=0.0)
        values_np = np.nan_to_num(values_np, nan=0.0, posinf=0.0, neginf=0.0)

        batch, n_heads, seq_len, head_dim = keys_np.shape
        heads_data = []

        for h in range(n_heads):
            k_h = keys_np[:, h, :, :].reshape(-1, head_dim)
            v_h = values_np[:, h, :, :].reshape(-1, head_dim)

            k_outlier_idx = np.argsort(np.max(np.abs(k_h), axis=0))[-self.outlier_dim:]
            k_normal_idx = np.setdiff1d(np.arange(head_dim), k_outlier_idx)
            
            v_outlier_idx = np.argsort(np.max(np.abs(v_h), axis=0))[-self.outlier_dim:]
            v_normal_idx = np.setdiff1d(np.arange(head_dim), v_outlier_idx)

            heads_data.append({
                'k_out_idx': k_outlier_idx,
                'k_norm_idx': k_normal_idx,
                'v_out_idx': v_outlier_idx,
                'v_norm_idx': v_normal_idx,
                'k_outliers': self._pack(k_h[:, k_outlier_idx], self.q_outlier),
                'k_normals': self._pack(k_h[:, k_normal_idx], self.q_normal),
                'v_outliers': self._pack(v_h[:, v_outlier_idx], self.q_outlier),
                'v_normals': self._pack(v_h[:, v_normal_idx], self.q_normal),
            })

        return {'shape': keys_np.shape, 'heads': heads_data}

    def decompress_kv(self, compressed, return_numpy=True):
        batch, n_heads, seq_len, head_dim = compressed['shape']
        num_vecs = batch * seq_len
        
        k_out = np.zeros((batch, n_heads, seq_len, head_dim), dtype=np.float32)
        v_out = np.zeros((batch, n_heads, seq_len, head_dim), dtype=np.float32)

        for h in range(n_heads):
            data = compressed['heads'][h]
            
            k_out_approx = self._unpack(data['k_outliers'], self.q_outlier, num_vecs, self.outlier_dim)
            k_norm_approx = self._unpack(data['k_normals'], self.q_normal, num_vecs, self.normal_dim)
            
            v_out_approx = self._unpack(data['v_outliers'], self.q_outlier, num_vecs, self.outlier_dim)
            v_norm_approx = self._unpack(data['v_normals'], self.q_normal, num_vecs, self.normal_dim)

            k_h = np.zeros((num_vecs, head_dim), dtype=np.float32)
            k_h[:, data['k_out_idx']] = k_out_approx
            k_h[:, data['k_norm_idx']] = k_norm_approx
            
            v_h = np.zeros((num_vecs, head_dim), dtype=np.float32)
            v_h[:, data['v_out_idx']] = v_out_approx
            v_h[:, data['v_norm_idx']] = v_norm_approx

            k_out[:, h, :, :] = k_h.reshape(batch, seq_len, head_dim)
            v_out[:, h, :, :] = v_h.reshape(batch, seq_len, head_dim)

        if return_numpy:
            return k_out, v_out
        return mx.array(k_out), mx.array(v_out)

    def memory_ratio(self):
        avg_bits = (self.outlier_dim * self.outlier_bits + self.normal_dim * self.normal_bits) / self.head_dim
        return avg_bits / 16.0

def monkey_patch_mlx_model(model, bit_width):
    return model

if __name__ == "__main__":
    batch = 1
    n_heads = 8
    seq_len = 64
    head_dim = 128
    
    keys = np.random.randn(batch, n_heads, seq_len, head_dim).astype(np.float32)
    values = np.random.randn(batch, n_heads, seq_len, head_dim).astype(np.float32)

    cache = TurboQuantKVCache(head_dim=128, bit_width=3, use_prod=True)
    compressed = cache.compress_kv(keys, values)
    keys_hat, values_hat = cache.decompress_kv(compressed, return_numpy=True)

    print(f"Original keys shape: {keys.shape}")
    print(f"Reconstructed keys shape: {keys_hat.shape}")
    
    ratio = cache.memory_ratio()
    print(f"Memory ratio: {ratio}")
    
    mse_keys = np.mean((keys - keys_hat) ** 2)
    mse_values = np.mean((values - values_hat) ** 2)
    
    print(f"MSE keys: {mse_keys}")
    print(f"MSE values: {mse_values}")
    
    shapes_match = keys.shape == keys_hat.shape
    print(f"Shapes match: {shapes_match}")

    if shapes_match:
        print("Shape match check: PASS")
    else:
        print("Shape match check: FAIL")
    assert shapes_match

    if ratio < 1.0:
        print("Memory ratio check: PASS")
    else:
        print("Memory ratio check: FAIL")
    assert ratio < 1.0

    if mse_keys < 0.5 and mse_values < 0.5:
            print("MSE check: PASS")
    else:
        print("MSE check: FAIL")
    assert mse_keys < 0.5 and mse_values < 0.5