#!/usr/bin/env python3
"""
Active Learning Loop for Quantum ML Data Generation
====================================================

Simple loop:
1. Start with 100K Sobol initial data
2. Each round: score 10M random candidates, select top 100K, query Julia
3. Repeat for 9 rounds → 1M total samples

Usage:
    python scripts/active_learning/run_al.py \
        --output data/al_results \
        --acquisition ucb_prime \
        --n_rounds 9
"""

import argparse
import gc
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from models import SurrogateGP, SurrogateEnsemble, SurrogateSVM, SurrogateErrorPredictor, create_surrogate
from acquisition import ACQUISITION_FUNCTIONS


# =============================================================================
# Julia Oracle Interface
# =============================================================================

def init_julia(julia_path):
    """Initialize Julia and load Hamiltonian code."""
    from julia.api import Julia
    Julia(compiled_modules=False)
    from julia import Main
    Main.include(str(julia_path))
    return Main


def query_oracle(julia_main, h_i, J_ij, theta, n_qubits, edges):
    """Query Julia oracle for a single configuration."""
    Jij_dict = {edge: float(J_ij[k]) for k, edge in enumerate(edges)}
    Xi_raw, ZZij_raw = julia_main.oracle(Jij_dict, h_i, theta, n_qubits, edges)
    Xi = np.ravel(np.asarray(Xi_raw, dtype=np.float64))
    ZZij = np.ravel(np.asarray(ZZij_raw, dtype=np.float64))
    return Xi, ZZij


def query_oracle_batch(julia_main, candidates, n_qubits, edges, chunk_size=25000):
    """
    Query Julia oracle for a batch of candidates.
    
    Args:
        candidates: (N, n_qubits + n_edges) array of [h_i, J_ij]
        chunk_size: Process in chunks to allow GC
    
    Returns:
        X: (N, n_qubits + n_edges) array of [Xi, J_ij]  (ML input)
        y: (N, n_edges) array of ZZij                    (ML target)
    """
    n_samples = len(candidates)
    n_edges = len(edges)
    
    # Random theta for each sample
    rng = np.random.default_rng()
    thetas = rng.uniform(0, np.pi / 2, size=n_samples)
    
    all_Xi, all_ZZij = [], []
    start_time = time.time()
    
    for i in range(n_samples):
        h_i = np.array(candidates[i, :n_qubits], dtype=np.float64)
        J_ij = np.array(candidates[i, n_qubits:], dtype=np.float64)
        
        Xi, ZZij = query_oracle(julia_main, h_i, J_ij, thetas[i], n_qubits, edges)
        all_Xi.append(Xi)
        all_ZZij.append(ZZij)
        
        # Progress every 5%
        if (i + 1) % max(1, n_samples // 20) == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate / 60
            print(f"      {i+1:>8,}/{n_samples:,} ({100*(i+1)/n_samples:5.1f}%) | "
                  f"{rate:.1f}/s | ETA: {eta:.1f}min")
        
        # GC every chunk
        if (i + 1) % chunk_size == 0:
            gc.collect()
    
    Xi = np.vstack(all_Xi)
    ZZij = np.vstack(all_ZZij)
    
    # Build ML format: X = [Xi, J_ij], y = ZZij
    J_ij_all = candidates[:, n_qubits:]
    X = np.hstack([Xi, J_ij_all])
    y = ZZij
    
    return X, y


# =============================================================================
# Data Generation
# =============================================================================

def generate_sobol(n_samples, n_qubits, n_edges, seed=42):
    """Generate Sobol quasi-random samples: h_i in [0,1], J_ij in [-1,1]."""
    dim = n_qubits + n_edges
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    raw = sampler.random(n_samples)
    
    h_i = raw[:, :n_qubits]                    # [0, 1]
    J_ij = raw[:, n_qubits:] * 2 - 1           # [-1, 1]
    
    return np.hstack([h_i, J_ij])


def generate_random(n_samples, n_qubits, n_edges):
    """Generate uniform random samples: h_i in [0,1], J_ij in [-1,1]."""
    rng = np.random.default_rng()
    h_i = rng.uniform(0, 1, (n_samples, n_qubits))
    J_ij = rng.uniform(-1, 1, (n_samples, n_edges))
    return np.hstack([h_i, J_ij])


def generate_clipped(n_samples, n_qubits, n_edges, clip_range=1.5):
    """
    Generate clipped samples that concentrate probability at boundaries.
    
    Sample from wider range [-clip_range, clip_range] then clip to [-1, 1].
    This piles up probability mass at ±1.
    
    Example with clip_range=1.5:
        - 33% of J values end up at -1 (clipped from [-1.5, -1])
        - 33% of J values end up at +1 (clipped from [1, 1.5])
        - 33% of J values in interior (from [-1, 1])
    """
    rng = np.random.default_rng()
    h_i = rng.uniform(0, 1, (n_samples, n_qubits))
    J_ij_raw = rng.uniform(-clip_range, clip_range, (n_samples, n_edges))
    J_ij = np.clip(J_ij_raw, -1, 1)
    return np.hstack([h_i, J_ij])


# =============================================================================
# I/O
# =============================================================================

def save_data(X, y, output_dir, seed=42):
    """Save with 80/10/10 train/val/test split."""
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
    
    return n_train, n_val, n - n_train - n_val


def load_data(data_dir):
    """Load all data from a directory."""
    data_dir = Path(data_dir)
    X_all, y_all = [], []
    
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.pt"
        if path.exists():
            data = torch.load(path, weights_only=False)
            X_all.append(data["X"].numpy())
            y_all.append(data["y"].numpy())
    
    return np.vstack(X_all), np.vstack(y_all)


# =============================================================================
# Main Loop
# =============================================================================

def run(args):
    """Main active learning loop."""
    
    # Setup
    n_qubits = args.n_qubits
    edges = list(itertools.combinations(range(1, n_qubits + 1), 2))  # 1-indexed for Julia
    n_edges = len(edges)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    julia_path = Path(args.julia_file) if args.julia_file else \
                 Path(__file__).parent.parent.parent / "src" / "gene_transverse_field_Ising.jl"
    
    # Logging
    log_file = output_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    
    log("=" * 70)
    log("ACTIVE LEARNING FOR QUANTUM ML")
    log("=" * 70)
    log(f"  Output:        {output_dir}")
    log(f"  Initial:       {args.n_initial:,} Sobol samples")
    log(f"  Rounds:        {args.n_rounds}")
    log(f"  Candidates:    {args.n_candidates:,} per round ({args.candidate_method})")
    if args.candidate_method == "clipped":
        log(f"  Clip range:    {args.clip_range} (boundary prob: {100*(args.clip_range-1)/args.clip_range:.0f}%)")
    log(f"  Select:        {args.n_select:,} per round")
    log(f"  Acquisition:   {args.acquisition}")
    log(f"  Surrogate:     {args.surrogate}")
    if args.surrogate == "gp":
        log(f"  GP inducing:   {args.gp_inducing}")
        log(f"  GP epochs:     {args.gp_epochs}")
    elif args.surrogate == "ensemble":
        log(f"  Ensemble:      {args.ensemble_members} members")
        log(f"  Ens hidden:    {args.ensemble_hidden}")
        log(f"  Ens layers:    {args.ensemble_layers}")
        log(f"  Ens epochs:    {args.ensemble_epochs}")
    elif args.surrogate == "svm":
        log(f"  SVM members:   {args.svm_members}")
        log(f"  SVM C:         {args.svm_C}")
        log(f"  SVM epsilon:   {args.svm_epsilon}")
        log(f"  SVM kernel:    {args.svm_kernel}")
        log(f"  SVM max_samp:  {args.svm_max_samples:,}")
    elif args.surrogate == "error_predictor":
        log(f"  ErrPred path:  {args.error_predictor_path}")
    log(f"  Device:        {'cuda' if torch.cuda.is_available() else 'cpu'}")
    log("=" * 70)
    
    # Initialize Julia
    log("\n[INIT] Loading Julia...")
    t0 = time.time()
    julia_main = init_julia(julia_path)
    log(f"  Done in {time.time() - t0:.1f}s")
    
    # ==========================================================================
    # STEP 0: Generate initial Sobol data
    # ==========================================================================
    log(f"\n[ROUND 0] Generating {args.n_initial:,} initial Sobol samples...")
    
    t0 = time.time()
    initial_candidates = generate_sobol(args.n_initial, n_qubits, n_edges, seed=args.seed)
    log(f"  Generated candidates in {time.time() - t0:.1f}s")
    
    log("  Querying Julia oracle...")
    t0 = time.time()
    X_train, y_train = query_oracle_batch(julia_main, initial_candidates, n_qubits, edges)
    log(f"  Oracle complete in {(time.time() - t0)/60:.1f}min")
    
    # Save initial data
    init_dir = output_dir / "round_0"
    n_tr, n_va, n_te = save_data(X_train, y_train, init_dir, seed=args.seed)
    log(f"  Saved to {init_dir}: train={n_tr:,}, val={n_va:,}, test={n_te:,}")
    
    # Get acquisition function
    acq_fn = ACQUISITION_FUNCTIONS[args.acquisition]
    
    # ==========================================================================
    # ACTIVE LEARNING LOOP
    # ==========================================================================
    for round_num in range(1, args.n_rounds + 1):
        log(f"\n{'='*70}")
        log(f"[ROUND {round_num}/{args.n_rounds}] Training set: {len(X_train):,} samples")
        log("=" * 70)
        round_start = time.time()
        
        # ----------------------------------------------------------------------
        # Step 1: Train surrogate
        # ----------------------------------------------------------------------
        if args.surrogate == "gp":
            surrogate_name = "SVGP"
        elif args.surrogate == "ensemble":
            surrogate_name = f"GNN Ensemble ({args.ensemble_members})"
        elif args.surrogate == "svm":
            surrogate_name = f"SVM Ensemble ({args.svm_members})"
        else:
            surrogate_name = "Error Predictor"
        
        log(f"\n  [1/4] Training {surrogate_name} surrogate...")
        t0 = time.time()
        
        surrogate = create_surrogate(
            surrogate_type=args.surrogate,
            # GP params
            num_inducing=args.gp_inducing,
            num_epochs=args.gp_epochs if args.surrogate == "gp" else args.ensemble_epochs,
            num_tasks=n_edges,
            # Ensemble params
            n_members=args.ensemble_members if args.surrogate == "ensemble" else args.svm_members,
            hidden_dim=args.ensemble_hidden,
            num_layers=args.ensemble_layers,
            batch_size=args.ensemble_batch,
            lr=args.ensemble_lr,
            n_qubits=n_qubits,
            # SVM params
            svm_C=args.svm_C,
            svm_epsilon=args.svm_epsilon,
            svm_kernel=args.svm_kernel,
            svm_gamma=args.svm_gamma,
            svm_max_samples=args.svm_max_samples,
            # Error predictor params
            error_predictor_path=args.error_predictor_path,
        )
        surrogate.fit(X_train, y_train, verbose=True)
        log(f"  Surrogate trained in {(time.time() - t0)/60:.1f}min")
        
        # ----------------------------------------------------------------------
        # Step 2: Generate candidate pool
        # ----------------------------------------------------------------------
        log(f"\n  [2/4] Generating {args.n_candidates:,} candidates ({args.candidate_method})...")
        t0 = time.time()
        
        if args.candidate_method == "random":
            candidates = generate_random(args.n_candidates, n_qubits, n_edges)
        elif args.candidate_method == "clipped":
            candidates = generate_clipped(args.n_candidates, n_qubits, n_edges, args.clip_range)
        else:
            candidates = generate_random(args.n_candidates, n_qubits, n_edges)
        
        log(f"  Generated in {time.time() - t0:.1f}s")
        
        # ----------------------------------------------------------------------
        # Step 3: Score candidates with acquisition function
        # ----------------------------------------------------------------------
        log(f"\n  [3/4] Scoring with {args.acquisition}...")
        t0 = time.time()
        
        candidates_tensor = torch.tensor(candidates, dtype=torch.float32)
        
        if args.acquisition == "uncertainty":
            scores = acq_fn(surrogate, candidates_tensor)
        else:
            scores = acq_fn(surrogate, candidates_tensor, beta=args.beta)
        
        log(f"  Scored in {(time.time() - t0)/60:.1f}min")
        log(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        log(f"  Score mean:  {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Select top-k
        _, top_idx = torch.topk(scores, args.n_select)
        selected = candidates[top_idx.numpy()]
        log(f"  Selected top {args.n_select:,} candidates")
        
        # Free memory
        del surrogate, candidates, candidates_tensor, scores
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # ----------------------------------------------------------------------
        # Step 4: Query oracle for selected candidates
        # ----------------------------------------------------------------------
        log(f"\n  [4/4] Querying Julia oracle for {args.n_select:,} samples...")
        t0 = time.time()
        
        X_new, y_new = query_oracle_batch(julia_main, selected, n_qubits, edges)
        log(f"  Oracle complete in {(time.time() - t0)/60:.1f}min")
        
        # Add to training set
        X_train = np.vstack([X_train, X_new])
        y_train = np.vstack([y_train, y_new])
        
        # Save round data
        round_dir = output_dir / f"round_{round_num}"
        n_tr, n_va, n_te = save_data(X_new, y_new, round_dir, seed=args.seed + round_num)
        log(f"  Saved to {round_dir}: train={n_tr:,}, val={n_va:,}, test={n_te:,}")
        
        round_elapsed = (time.time() - round_start) / 60
        log(f"\n  Round {round_num} complete in {round_elapsed:.1f}min")
        log(f"  Total training set: {len(X_train):,} samples")
        
        gc.collect()
    
    # ==========================================================================
    # SAVE FINAL COMBINED DATASET
    # ==========================================================================
    log(f"\n{'='*70}")
    log("SAVING FINAL COMBINED DATASET")
    log("=" * 70)
    
    final_dir = output_dir / "final"
    n_tr, n_va, n_te = save_data(X_train, y_train, final_dir, seed=args.seed)
    log(f"Saved {len(X_train):,} samples to {final_dir}")
    log(f"  train: {n_tr:,}, val: {n_va:,}, test: {n_te:,}")
    
    log(f"\n{'='*70}")
    log("ACTIVE LEARNING COMPLETE")
    log("=" * 70)
    log(f"Total samples: {len(X_train):,}")
    log(f"Output:        {output_dir}")
    log(f"Log:           {log_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Active Learning for Quantum ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Output
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--julia_file", default=None, help="Path to Julia oracle")
    
    # Data sizes
    parser.add_argument("--n_initial", type=int, default=100_000,
                        help="Initial Sobol samples")
    parser.add_argument("--n_rounds", type=int, default=9,
                        help="Number of AL rounds")
    parser.add_argument("--n_candidates", type=int, default=10_000_000,
                        help="Candidates to score per round")
    parser.add_argument("--n_select", type=int, default=100_000,
                        help="Samples to select per round")
    
    # Candidate generation
    parser.add_argument("--candidate_method", choices=["random", "clipped"],
                        default="random", help="Candidate generation method")
    parser.add_argument("--clip_range", type=float, default=1.5,
                        help="Range for clipped method (1.5 = 33%% at boundaries)")
    
    # Acquisition
    parser.add_argument("--acquisition", choices=["ucb", "ucb_prime", "uncertainty"],
                        default="uncertainty", help="Acquisition function")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Exploration parameter for UCB/UCB'")
    
    # Surrogate choice
    parser.add_argument("--surrogate", choices=["gp", "ensemble", "svm", "error_predictor"],
                        default="ensemble", help="Surrogate model type")
    
    # GP params (only used if surrogate=gp)
    parser.add_argument("--gp_inducing", type=int, default=500,
                        help="Number of inducing points (GP only)")
    parser.add_argument("--gp_epochs", type=int, default=50,
                        help="SVGP training epochs (GP only)")
    
    # Ensemble params (only used if surrogate=ensemble)
    parser.add_argument("--ensemble_members", type=int, default=5,
                        help="Number of GNN ensemble members")
    parser.add_argument("--ensemble_hidden", type=int, default=64,
                        help="Hidden dimension for ensemble GNNs")
    parser.add_argument("--ensemble_layers", type=int, default=3,
                        help="Number of message passing layers")
    parser.add_argument("--ensemble_epochs", type=int, default=30,
                        help="Training epochs per ensemble member")
    parser.add_argument("--ensemble_batch", type=int, default=256,
                        help="Batch size for ensemble training")
    parser.add_argument("--ensemble_lr", type=float, default=1e-3,
                        help="Learning rate for ensemble training")
    
    # SVM params (only used if surrogate=svm)
    parser.add_argument("--svm_members", type=int, default=5,
                        help="Number of SVM ensemble members")
    parser.add_argument("--svm_C", type=float, default=1.0,
                        help="SVM regularization parameter")
    parser.add_argument("--svm_epsilon", type=float, default=0.1,
                        help="Epsilon in epsilon-SVR model")
    parser.add_argument("--svm_kernel", choices=["rbf", "linear", "poly"],
                        default="rbf", help="SVM kernel type")
    parser.add_argument("--svm_gamma", type=str, default="scale",
                        help="Kernel coefficient ('scale', 'auto', or float)")
    parser.add_argument("--svm_max_samples", type=int, default=50000,
                        help="Max samples per SVM (subsamples if larger)")
    
    # Error predictor params (only used if surrogate=error_predictor)
    parser.add_argument("--error_predictor_path", type=str, default=None,
                        help="Path to pre-trained error predictor (required for error_predictor surrogate)")
    
    # Other
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
