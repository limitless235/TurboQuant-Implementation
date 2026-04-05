import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import time
import numpy as np
from datasets import load_dataset
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from integration.mlx_kv_hook import TurboQuantKVCache
from datetime import datetime

def build_haystack(background_text, context_length, depth_percent, needle):
    words = background_text.split()
    if len(words) < context_length:
        words = (words * (context_length // len(words) + 1))[:context_length]
    else:
        words = words[:context_length]
    
    insert_index = int((len(words) - 1) * depth_percent)
    words.insert(insert_index, needle)
    return " ".join(words)

def run_evaluation(mode, args, background_text):
    print(f"\n{'='*50}\nStarting evaluation mode: {mode.upper()}\n{'='*50}")

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model {args.model} into memory...")
    model, tokenizer = load(args.model)

    out_dir = os.path.join(args.output_dir, mode)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        predictions_path = os.path.join(out_dir, "predictions.json")
        pred_file = open(predictions_path, "w")

    needle = "The special secret code is 'BANANA-77'."
    question = "What is the special secret code?"
    expected_answer = "BANANA-77"

    contexts = [1000, 2000, 4000, 8000]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    if args.dry_run:
        contexts = [1000]
        depths = [0.5]

    total_runs = len(contexts) * len(depths)
    current_run = 0
    total_time = 0.0
    results_summary = []

    print(f"\nStarting Needle In A Haystack generation for {total_runs} configurations...\n")

    for ctx_len in contexts:
        for depth in depths:
            current_run += 1
            haystack = build_haystack(background_text, ctx_len, depth, needle)
            prompt = f"{haystack}\n\nQuestion: {question}\nAnswer:"
            
            start_time = time.time()
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            time_seconds = time.time() - start_time
            total_time += time_seconds
            success = expected_answer in response
            
            mse_report = ""
            
            if mode == "turboquant":
                prompt_tokens = mx.array(tokenizer.encode(prompt))[None]
                prompt_cache = make_prompt_cache(model)
                _ = model(prompt_tokens, cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                
                head_dim = prompt_cache[0].keys.shape[-1]
                compressor = TurboQuantKVCache(head_dim=head_dim, bit_width=args.bit_width, use_prod=True)
                
                layer_mses = []
                for layer_idx, c in enumerate(prompt_cache):
                    compressed = compressor.compress_kv(c.keys, c.values)
                    k_hat, v_hat = compressor.decompress_kv(compressed, return_numpy=True)
                    
                    k_orig = np.array(c.keys.astype(mx.float32))
                    v_orig = np.array(c.keys.astype(mx.float32))
                    mse_k = np.mean((k_orig - k_hat) ** 2)
                    mse_v = np.mean((v_orig - v_hat) ** 2)
                    layer_mses.append((mse_k + mse_v) / 2.0)
                
                avg_mse = np.mean(layer_mses)
                mse_report = f" | Cache MSE (3-bit): {avg_mse:.4f}"

            print(f"[{current_run}/{total_runs}] Context: {ctx_len} words, Depth: {depth*100:^4.0f}% -> {'PASS' if success else 'FAIL'} ({time_seconds:.2f}s){mse_report}")

            if not args.dry_run:
                pred_entry = {
                    "context_length": ctx_len,
                    "depth": depth,
                    "success": success,
                    "model_response": response.strip(),
                    "time_seconds": time_seconds,
                    "cache_mse": float(avg_mse) if mode == "turboquant" else None
                }
                pred_file.write(json.dumps(pred_entry) + "\n")
                pred_file.flush()
                results_summary.append(pred_entry)

    if not args.dry_run:
        pred_file.close()
        pass_count = sum(1 for r in results_summary if r["success"])
        
        summary = {
            "mode": mode,
            "bit_width": args.bit_width if mode == "turboquant" else None,
            "total_runs": total_runs,
            "pass_count": pass_count,
            "accuracy": pass_count / total_runs if total_runs > 0 else 0,
            "total_time_seconds": total_time,
            "model": args.model,
            "timestamp": datetime.now().isoformat()
        }

        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "turboquant", "both"], default="turboquant")
    parser.add_argument("--bit_width", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results_niah")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    print("Downloading/Loading background text for Haystack...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    background_text = " ".join([row["text"] for row in dataset if row["text"].strip()])

    modes = ["baseline", "turboquant"] if args.mode == "both" else [args.mode]

    for mode in modes:
        run_evaluation(mode, args, background_text)

if __name__ == "__main__":
    main()