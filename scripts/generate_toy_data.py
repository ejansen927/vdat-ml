#!/usr/bin/env python3
""" Data for Toy problem, fixed J's.
Generate data at fixed J=[1,1,1,1,1,1] corner to test if model can learn it.
"""

import argparse
import itertools
import numpy as np
import torch
from pathlib import Path
import time

def init_julia(julia_path):
    from julia.api import Julia
    Julia(compiled_modules=False)
    from julia import Main
    Main.include(str(julia_path))
    return Main

def generate_toy_data(julia_main, n_samples, n_qubits=4, j_value=1.0):
    edges = list(itertools.combinations(range(1, n_qubits + 1), 2))
    n_edges = len(edges)
    
    rng = np.random.default_rng(42)
    
    h_i_all = rng.uniform(0, 1, (n_samples, n_qubits))
    theta_all = rng.uniform(0, np.pi / 2, n_samples)
    J_ij_fixed = np.full((n_samples, n_edges), j_value)
    
    all_Xi, all_ZZij = [], []
    
    print(f"Generating {n_samples} samples at J=[{j_value}]*6...")
    t0 = time.time()
    
    for i in range(n_samples):
        h_i = np.array(h_i_all[i], dtype=np.float64)
        J_ij = np.array(J_ij_fixed[i], dtype=np.float64)
        theta = float(theta_all[i])
        
        Jij_dict = {edge: float(J_ij[k]) for k, edge in enumerate(edges)}
        Xi_raw, ZZij_raw = julia_main.oracle(Jij_dict, h_i, theta, n_qubits, edges)
        
        Xi = np.ravel(np.asarray(Xi_raw, dtype=np.float64))
        ZZij = np.ravel(np.asarray(ZZij_raw, dtype=np.float64))
        
        all_Xi.append(Xi)
        all_ZZij.append(ZZij)
        
        if (i + 1) % max(1, n_samples // 10) == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%) - {elapsed:.1f}s")
    
    Xi = np.vstack(all_Xi)
    ZZij = np.vstack(all_ZZij)
    
    X = np.hstack([Xi, J_ij_fixed])
    y = ZZij
    
    return X, y

def save_data(X, y, output_dir, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    
    splits = {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }
    
    for name, idx in splits.items():
        torch.save({
            "X": torch.tensor(X[idx], dtype=torch.float32),
            "y": torch.tensor(y[idx], dtype=torch.float32),
        }, output_dir / f"{name}.pt")
        print(f"  {name}: {len(idx)} samples")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--j_value", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="data/toy_test")
    parser.add_argument("--julia_file", type=str, default="src/gene_transverse_field_Ising.jl")
    args = parser.parse_args()
    
    julia_main = init_julia(args.julia_file)
    X, y = generate_toy_data(julia_main, args.n_samples, j_value=args.j_value)
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"y mean:  {y.mean():.4f}")
    
    save_data(X, y, args.output)
    print(f"\nSaved to {args.output}/")

if __name__ == "__main__":
    main()
