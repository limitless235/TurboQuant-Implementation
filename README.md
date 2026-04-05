#  TurboQuant MLX Implementation

An independent, from-scratch implementation of **TurboQuant** built specifically for Apple Silicon using the **MLX framework**. 

This repository provides an online, near-optimal vector quantization strategy for Large Language Model (LLM) KV caches. By leveraging random coordinate rotations and outlier-aware channel splitting, this implementation compresses the KV cache to **15.6% of its original size** (an effective 2.5 bits per channel) with **zero degradation** in long-context retrieval tasks.

---

## Background: The Outlier Problem

As context windows in LLMs expand, the KV Cache becomes a severe memory bottleneck. Standard quantization techniques fail because LLM activation data is highly "spiky." While most channel values are near zero, specific attention sinks hold massive outlier values. Uniformly squashing these vectors destroys the structural geometry, resulting in severe amnesia for the model.

**The Solution:**
1. **Random Rotation:** Input vectors are rotated to spread the values into a predictable, smooth distribution.
2. **Channel Splitting:** The data is dynamically split. The top 32 outlier channels receive 4-bit precision to safeguard their massive values, while the remaining 96 normal channels are aggressively compressed to 2-bit precision. 

In benchmarking with Qwen2.5-1.5B, this approach successfully passed the Needle In A Haystack test at an 8,000-word context limit with 100% accuracy.

---

##  Repository Structure & File Explanations

###  Core Algorithms (`/core`)
This directory contains the pure mathematical primitives of the TurboQuant paper, entirely framework-agnostic.

* **`quantize_mse.py`**: Implements the base `TurboQuantMSE` algorithm. It applies random rotation and quantizes coordinates using the Max-Lloyd algorithm to minimize Mean Squared Error.
* **`quantize_prod.py`**: Implements `TurboQuantProd`, optimized for inner-product estimation. Essential for attention mechanisms where vector geometry and angles matter more than pure coordinate accuracy.
* **`qjl.py`**: Implements the Quantized Johnson-Lindenstrauss (QJL) transform. This applies a 1-bit quantizer to the residual error of the MSE quantizer, correcting inherent biases and ensuring low-distortion inner products.
* **`rotation.py`**: Handles the generation and application of random orthogonal rotation matrices (via QR decomposition) to uniformly distribute data across dimensions.
* **`codebook.py`**: Utility functions to load the precomputed Voronoi tessellation centroids for optimal scalar quantization.

### MLX Integration (`/integration`)
This directory bridges the mathematical core with Apple's MLX library to actively compress the memory of running models.

* **`mlx_kv_hook.py`**: The heart of the integration. 
  * Contains the `TurboQuantKVCache` class which manages the packing and unpacking of MLX arrays.
  * Implements the **Channel Splitting** logic, routing the 32 highest-magnitude channels to 4-bit quantization and the rest to 2-bit.
  * Features the `QuantizedAttentionWrapper` to dynamically monkey-patch MLX LLMs, hijacking their forward pass to simulate quantization noise directly in the attention layer.

### Benchmarking & Testing (`/bench`)
Scripts to validate the mathematical boundaries and practical performance of the implementation.

* **`swebench_runner.py`**: Executes the Needle In A Haystack benchmark. It injects a hidden phrase into large context windows (up to 8,000 tokens), compresses the active cache using our MLX hooks, and forces the model to retrieve the phrase to test for context degradation.
* **`distortion.py`**: A mathematical validation script that computes the absolute Mean Squared Error (MSE) and variance distributions against the theoretical lower bounds established in the TurboQuant paper.
* **`figures/`**: Contains generated plots (e.g., `distortion_validation.png`) visualizing the error rates across different bit-widths.

### Results (`/results_niah`)
Stores the raw telemetry from benchmark runs.
* **`baseline/`**: Output data (`predictions.json`, `summary.json`) from the full-precision, uncompressed LLM.
* **`turboquant/`**: Output data from the model running with the 2.5-bit effective TurboQuant cache.

###  Root Assets
* **`codebook_*.pkl` / `codebooks.npz`**: Precomputed 1D k-means centroids for various dimensions (128, 1536) and bit-widths. Caching these offline eliminates online calibration overhead, allowing the quantizer to run instantly during token generation.
* **`requirements.txt`**: Project dependencies (`mlx`, `mlx-lm`, `numpy`, `datasets`).

---

## Getting Started

### Prerequisites
* Apple Silicon Mac (M1/M2/M3/M4)
* Python 3.12+

### Installation
```bash
git clone [https://github.com/yourusername/TurboQuant-Implementation.git](https://github.com/yourusername/TurboQuant-Implementation.git)
cd TurboQuant-Implementation
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
