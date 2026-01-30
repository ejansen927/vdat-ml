#!/usr/bin/env python3
"""
Data Generation Script
======================

Generates quantum data by calling Julia oracle and saves in .pt format
ready for training. Automatically creates the corresponding Hydra config.

Usage (run from project root):
    python scripts/generate_data.py --dist random --n_samples 1000 --n_qubits 4
    
Output:
    data/{dist}_{n_qubits}q/
    ├── train.pt
    ├── val.pt
    ├── test.pt
    └── metadata.yaml
    
    configs/data/{dist}_{n_qubits}q.yaml        (tensor config for MLP)
    configs/data/{dist}_{n_qubits}q_graph.yaml  (graph config for GNN, if --graph_config)
"""

import argparse
import itertools
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

import numpy as np
import torch


# =============================================================================
# Path Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_JULIA_FILE = PROJECT_ROOT / "src" / "gene_transverse_field_Ising.jl"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "data"


# =============================================================================
# Julia Interface
# =============================================================================

def init_julia(hamiltonian_path: Path):
    """Initialize Julia and load the Hamiltonian code."""
    print(f"  Julia file: {hamiltonian_path}")
    
    if not hamiltonian_path.exists():
        raise FileNotFoundError(
            f"Julia file not found: {hamiltonian_path}\n"
            f"Expected at: {hamiltonian_path.resolve()}"
        )
    
    from julia.api import Julia
    Julia(compiled_modules=False)
    from julia import Main
    
    Main.include(str(hamiltonian_path))
    return Main


def call_oracle(
    Main,
    Jij_dict: Dict[Tuple[int, int], float],
    h_i: np.ndarray,
    theta: float,
    n_qubits: int,
    edges: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Call Julia oracle to compute Xi and ZZij."""
    Xi_raw, ZZij_raw = Main.oracle(Jij_dict, h_i, theta, n_qubits, edges)
    Xi = np.ravel(np.asarray(Xi_raw, dtype=np.float64))
    ZZij = np.ravel(np.asarray(ZZij_raw, dtype=np.float64))
    return Xi, ZZij


# =============================================================================
# Distribution Samplers
# =============================================================================

def sample_random(
    n_qubits: int,
    n_edges: int,
    rng: np.random.Generator,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Sample from uniform random distribution.
    
    h_i:   [0, 1]
    J_ij:  [-1, 1]
    theta: [0, π/2]
    """
    h_i = rng.uniform(0.0, 1.0, size=n_qubits)
    J_ij = rng.uniform(-1.0, 1.0, size=n_edges)
    theta = rng.uniform(0.0, np.pi / 2)
    return h_i, J_ij, theta


def sample_gaussian(
    n_qubits: int,
    n_edges: int,
    rng: np.random.Generator,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """TODO: Implement truncated Gaussian sampling."""
    raise NotImplementedError(
        "Gaussian distribution not yet implemented.\n"
        "Add your truncated normal logic here."
    )


DISTRIBUTIONS = {
    "random": sample_random,
    "gaussian": sample_gaussian,
    "sobol": None,  # Handled directly in generate_data
    "hybrid": None,  # Handled directly in generate_data
    "sweep": None,   # Handled directly in generate_data (theta sweep)
}


def generate_sweep_samples(
    n_base: int,
    n_theta: int,
    n_qubits: int,
    n_edges: int,
    base_dist: str = "random",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sweep samples: n_base [h_i, J_ij] points × n_theta theta values.
    
    Instead of randomly sampling theta, we systematically sweep it from 0 to π/2.
    This helps the model learn the theta dependence more systematically.
    
    Total samples = n_base × n_theta
    
    Args:
        n_base: Number of base [h_i, J_ij] configurations
        n_theta: Number of theta values to sweep (equally spaced in [0, π/2])
        n_qubits: Number of qubits
        n_edges: Number of edges
        base_dist: Distribution for [h_i, J_ij] ('random' or 'sobol')
        seed: Random seed
    
    Returns:
        h_i: (n_base * n_theta, n_qubits)
        J_ij: (n_base * n_theta, n_edges)
        theta: (n_base * n_theta,)
    """
    from scipy.stats import qmc
    
    rng = np.random.default_rng(seed)
    n_total = n_base * n_theta
    
    # Generate base [h_i, J_ij] samples
    if base_dist == "sobol":
        dim = n_qubits + n_edges
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        raw = sampler.random(n_base)
        base_h_i = raw[:, :n_qubits]  # [0, 1]
        base_J_ij = raw[:, n_qubits:] * 2 - 1  # [-1, 1]
    else:  # random
        base_h_i = rng.uniform(0.0, 1.0, size=(n_base, n_qubits))
        base_J_ij = rng.uniform(-1.0, 1.0, size=(n_base, n_edges))
    
    # Create theta sweep values (equally spaced from 0 to π/2, inclusive)
    theta_values = np.linspace(0, np.pi / 2, n_theta)
    
    # Expand: each base sample gets all theta values
    # Result: for each (h_i, J_ij) pair, we have n_theta samples with different theta
    all_h_i = np.repeat(base_h_i, n_theta, axis=0)      # (n_total, n_qubits)
    all_J_ij = np.repeat(base_J_ij, n_theta, axis=0)    # (n_total, n_edges)
    all_theta = np.tile(theta_values, n_base)           # (n_total,)
    
    # Shuffle to avoid having all same-config samples together during training
    perm = rng.permutation(n_total)
    
    return all_h_i[perm], all_J_ij[perm], all_theta[perm]


def generate_stratified_samples(
    n_samples: int,
    n_qubits: int,
    n_edges: int,
    boundary_width: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate stratified samples with equal representation per boundary stratum.
    
    Strata are defined by "how many dimensions are at the boundary".
    For 6 dimensions (edges), we have 7 strata (0 to 6 dims at boundary).
    
    This ensures corners (all dims at boundary) get as many samples as
    the interior (no dims at boundary), despite having ~10^6x smaller volume.
    
    Args:
        n_samples: Total samples to generate
        n_qubits: Number of qubits (for h_i)
        n_edges: Number of edges (for J_ij, defines strata)
        boundary_width: Width of boundary region [0, bw] and [1-bw, 1]
        seed: Random seed
    
    Returns:
        h_i: (n_samples, n_qubits) in [0, 1]
        J_ij: (n_samples, n_edges) in [-1, 1]  
        theta: (n_samples,) in [0, π/2]
    """
    rng = np.random.default_rng(seed)
    
    n_strata = n_edges + 1  # 0 to n_edges dims at boundary
    samples_per_stratum = n_samples // n_strata
    remainder = n_samples % n_strata
    
    all_h_i = []
    all_J_ij = []
    all_theta = []
    
    for k in range(n_strata):
        # k dimensions will be at boundary
        n_this_stratum = samples_per_stratum + (1 if k < remainder else 0)
        
        for _ in range(n_this_stratum):
            # Sample h_i uniformly (not stratified, just J_ij)
            h_i = rng.uniform(0.0, 1.0, size=n_qubits)
            
            # Sample theta uniformly
            theta = rng.uniform(0.0, np.pi / 2)
            
            # For J_ij: k dims at boundary, (n_edges - k) in interior
            # Work in [0, 1] first, then transform to [-1, 1]
            J_raw = np.zeros(n_edges)
            
            # Choose which k dimensions are at boundary
            boundary_dims = rng.choice(n_edges, size=k, replace=False) if k > 0 else []
            interior_dims = [i for i in range(n_edges) if i not in boundary_dims]
            
            # Boundary dims: sample from [0, bw] or [1-bw, 1]
            for d in boundary_dims:
                if rng.random() < 0.5:
                    J_raw[d] = rng.uniform(0, boundary_width)
                else:
                    J_raw[d] = rng.uniform(1 - boundary_width, 1)
            
            # Interior dims: sample from [bw, 1-bw]
            for d in interior_dims:
                J_raw[d] = rng.uniform(boundary_width, 1 - boundary_width)
            
            # Transform J from [0, 1] to [-1, 1]
            J_ij = J_raw * 2 - 1
            
            all_h_i.append(h_i)
            all_J_ij.append(J_ij)
            all_theta.append(theta)
    
    return (
        np.array(all_h_i),
        np.array(all_J_ij),
        np.array(all_theta),
    )


def generate_hybrid_samples(
    n_samples: int,
    n_qubits: int,
    n_edges: int,
    sobol_fraction: float = 0.5,
    boundary_width: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate hybrid samples: mixture of Sobol + stratified boundary sampling.
    
    This combines:
    - Sobol quasi-random: uniform low-discrepancy coverage of full space
    - Stratified: equal samples per "number of dims at boundary" stratum
    
    The result is a mixture distribution:
        p(x) = sobol_fraction * p_sobol(x) + (1 - sobol_fraction) * p_stratified(x)
    
    Args:
        n_samples: Total samples
        n_qubits: Number of qubits
        n_edges: Number of edges
        sobol_fraction: Fraction from Sobol (default 0.5 = 50/50 mix)
        boundary_width: Width of boundary region for stratified part
        seed: Random seed
    
    Returns:
        h_i, J_ij, theta arrays
    """
    from scipy.stats import qmc
    
    n_sobol = int(n_samples * sobol_fraction)
    n_stratified = n_samples - n_sobol
    
    print(f"    Hybrid mixture: {n_sobol:,} Sobol + {n_stratified:,} stratified")
    
    # Part 1: Sobol samples
    dim = n_qubits + n_edges + 1
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    sobol_raw = sampler.random(n_sobol)
    
    sobol_h_i = sobol_raw[:, :n_qubits]  # [0, 1]
    sobol_J_ij = sobol_raw[:, n_qubits:n_qubits + n_edges] * 2 - 1  # [-1, 1]
    sobol_theta = sobol_raw[:, -1] * (np.pi / 2)  # [0, π/2]
    
    # Part 2: Stratified samples
    strat_h_i, strat_J_ij, strat_theta = generate_stratified_samples(
        n_stratified, n_qubits, n_edges, boundary_width, seed=seed + 1000
    )
    
    # Combine
    all_h_i = np.vstack([sobol_h_i, strat_h_i])
    all_J_ij = np.vstack([sobol_J_ij, strat_J_ij])
    all_theta = np.concatenate([sobol_theta, strat_theta])
    
    # Shuffle to mix Sobol and stratified samples
    rng = np.random.default_rng(seed + 2000)
    perm = rng.permutation(n_samples)
    
    return all_h_i[perm], all_J_ij[perm], all_theta[perm]


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(
    n_samples: int,
    n_qubits: int,
    distribution: str,
    julia_main,
    seed: int = 42,
    verbose: bool = True,
    chunk_size: int = None,
    sobol_fraction: float = 0.5,
    boundary_width: float = 0.1,
    sweep_n_base: int = None,
    sweep_n_theta: int = 10,
    sweep_base_dist: str = "random",
) -> Dict[str, np.ndarray]:
    """
    Generate quantum ML dataset.
    
    Args:
        n_samples: Total number of samples to generate
        n_qubits: Number of qubits
        distribution: Distribution name ('random', 'sobol', 'hybrid', 'sweep')
        julia_main: Initialized Julia Main module
        seed: Random seed
        verbose: Print progress
        chunk_size: If set, process Julia calls in chunks with GC between them
        sobol_fraction: For hybrid dist, fraction from Sobol (default 0.5)
        boundary_width: For hybrid dist, boundary region width (default 0.1)
        sweep_n_base: For sweep dist, number of base [h_i, J_ij] configs
        sweep_n_theta: For sweep dist, number of theta points (default 10)
        sweep_base_dist: For sweep dist, base distribution ('random' or 'sobol')
    
    Returns dict with:
        X:  ML input [Xi, Jij], shape (N, n_qubits + n_edges)
        y:  ML target [ZZij], shape (N, n_edges)
        A:  Algorithm input [hi, Jij, theta], shape (N, n_qubits + n_edges + 1)
    """
    import gc
    import time
    import tempfile
    
    rng = np.random.default_rng(seed)
    # Julia uses 1-indexed edges
    edges_julia = list(itertools.combinations(range(1, n_qubits + 1), 2))
    n_edges = len(edges_julia)
    
    # Determine chunk size for Julia processing
    if chunk_size is None:
        chunk_size = min(25000, n_samples)  # Default 25K chunks
    
    # Determine print frequency based on dataset size
    if n_samples <= 100:
        print_freq = 10
    elif n_samples <= 1000:
        print_freq = 100
    elif n_samples <= 10000:
        print_freq = 500
    elif n_samples <= 100000:
        print_freq = 1000
    else:
        print_freq = 5000
    
    # Create temp directory for intermediate results
    temp_dir = Path(tempfile.mkdtemp(prefix="vdat_gen_"))
    if verbose:
        print(f"  Temp directory: {temp_dir}")
    
    # =========================================================================
    # Stage 1: Pre-generate all input samples and save to disk
    # =========================================================================
    if verbose:
        print(f"\n  [Stage 1] Pre-generating {n_samples:,} input samples ({distribution})...")
    
    if distribution == "sobol":
        from scipy.stats import qmc
        
        dim = n_qubits + n_edges + 1  # h_i + J_ij + theta
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        raw = sampler.random(n_samples)  # float64 by default
        
        # Scale and save to disk immediately
        all_h_i = raw[:, :n_qubits]  # [0, 1]
        all_J_ij = raw[:, n_qubits:n_qubits + n_edges] * 2 - 1  # [-1, 1]
        all_theta = raw[:, -1] * (np.pi / 2)  # [0, π/2]
        
        np.save(temp_dir / "h_i.npy", all_h_i)
        np.save(temp_dir / "J_ij.npy", all_J_ij)
        np.save(temp_dir / "theta.npy", all_theta)
        
        del raw, all_h_i, all_J_ij, all_theta
        gc.collect()
        
        if verbose:
            print(f"    Sobol samples saved to disk")
    
    elif distribution == "random":
        all_h_i = rng.uniform(0.0, 1.0, size=(n_samples, n_qubits))
        all_J_ij = rng.uniform(-1.0, 1.0, size=(n_samples, n_edges))
        all_theta = rng.uniform(0.0, np.pi / 2, size=n_samples)
        
        np.save(temp_dir / "h_i.npy", all_h_i)
        np.save(temp_dir / "J_ij.npy", all_J_ij)
        np.save(temp_dir / "theta.npy", all_theta)
        
        del all_h_i, all_J_ij, all_theta
        gc.collect()
        
        if verbose:
            print(f"    Random samples saved to disk")
    
    elif distribution == "hybrid":
        # Mixture of Sobol + stratified boundary sampling
        all_h_i, all_J_ij, all_theta = generate_hybrid_samples(
            n_samples=n_samples,
            n_qubits=n_qubits,
            n_edges=n_edges,
            sobol_fraction=sobol_fraction,
            boundary_width=boundary_width,
            seed=seed,
        )
        
        np.save(temp_dir / "h_i.npy", all_h_i)
        np.save(temp_dir / "J_ij.npy", all_J_ij)
        np.save(temp_dir / "theta.npy", all_theta)
        
        del all_h_i, all_J_ij, all_theta
        gc.collect()
        
        if verbose:
            print(f"    Hybrid (Sobol + stratified) samples saved to disk")
    
    elif distribution == "sweep":
        # Theta sweep: n_base configs × n_theta sweep points
        all_h_i, all_J_ij, all_theta = generate_sweep_samples(
            n_base=sweep_n_base,
            n_theta=sweep_n_theta,
            n_qubits=n_qubits,
            n_edges=n_edges,
            base_dist=sweep_base_dist,
            seed=seed,
        )
        
        # Verify we got the right number
        assert len(all_h_i) == n_samples, f"Sweep generated {len(all_h_i)} samples, expected {n_samples}"
        
        np.save(temp_dir / "h_i.npy", all_h_i)
        np.save(temp_dir / "J_ij.npy", all_J_ij)
        np.save(temp_dir / "theta.npy", all_theta)
        
        del all_h_i, all_J_ij, all_theta
        gc.collect()
        
        if verbose:
            print(f"    Sweep ({sweep_n_base:,} configs × {sweep_n_theta} theta) samples saved to disk")
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Choose from ['random', 'sobol', 'hybrid', 'sweep']")
    
    # =========================================================================
    # Stage 2: Process through Julia in chunks, saving each chunk to disk
    # =========================================================================
    if verbose:
        print(f"\n  [Stage 2] Processing through Julia (chunk_size={chunk_size:,})...")
    
    # Load inputs via memory mapping (read-only, doesn't load into RAM)
    all_h_i = np.load(temp_dir / "h_i.npy", mmap_mode='r')
    all_J_ij = np.load(temp_dir / "J_ij.npy", mmap_mode='r')
    all_theta = np.load(temp_dir / "theta.npy", mmap_mode='r')
    
    samples_processed = 0
    chunk_num = 0
    total_start_time = time.time()
    chunk_files = []
    
    while samples_processed < n_samples:
        chunk_num += 1
        chunk_start = samples_processed
        chunk_end = min(samples_processed + chunk_size, n_samples)
        chunk_n = chunk_end - chunk_start
        
        if verbose:
            print(f"\n    [Chunk {chunk_num}] Samples {chunk_start + 1:,} to {chunk_end:,}")
        
        chunk_start_time = time.time()
        
        # Process this chunk
        Xi_chunk = []
        ZZij_chunk = []
        
        for i in range(chunk_start, chunk_end):
            if verbose and (i + 1) % print_freq == 0:
                elapsed = time.time() - total_start_time
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate if rate > 0 else 0
                pct = 100 * (i + 1) / n_samples
                print(f"      {i + 1:>8,} / {n_samples:,} ({pct:5.1f}%) | "
                      f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f} min")
            
            # Extract sample - copy from mmap and ensure float64
            h_i = np.array(all_h_i[i], dtype=np.float64)
            J_ij = np.array(all_J_ij[i], dtype=np.float64)
            theta = float(all_theta[i])
            
            # Call Julia oracle
            Jij_dict = {edge: float(J_ij[k]) for k, edge in enumerate(edges_julia)}
            Xi, ZZij = call_oracle(julia_main, Jij_dict, h_i, theta, n_qubits, edges_julia)
            
            Xi_chunk.append(Xi)
            ZZij_chunk.append(ZZij)
        
        # Save chunk to disk
        chunk_file = temp_dir / f"chunk_{chunk_num:04d}.npz"
        np.savez(chunk_file, 
                 Xi=np.vstack(Xi_chunk), 
                 ZZij=np.vstack(ZZij_chunk))
        chunk_files.append(chunk_file)
        
        # Clear chunk from memory
        del Xi_chunk, ZZij_chunk
        
        samples_processed = chunk_end
        
        # Chunk summary and aggressive GC
        if verbose:
            chunk_elapsed = time.time() - chunk_start_time
            chunk_rate = chunk_n / chunk_elapsed if chunk_elapsed > 0 else 0
            print(f"      Chunk saved: {chunk_file.name} | {chunk_rate:.1f}/sec")
            print(f"      Running GC...")
        
        gc.collect()
    
    # =========================================================================
    # Stage 3: Load all chunks and combine
    # =========================================================================
    if verbose:
        total_elapsed = time.time() - total_start_time
        print(f"\n  Julia complete: {n_samples:,} samples in {total_elapsed:.1f}s ({n_samples/total_elapsed:.1f}/sec)")
        print(f"\n  [Stage 3] Combining {len(chunk_files)} chunks...")
    
    # Reload input arrays (need them for final output)
    all_h_i = np.load(temp_dir / "h_i.npy")
    all_J_ij = np.load(temp_dir / "J_ij.npy")
    all_theta = np.load(temp_dir / "theta.npy")
    
    # Load and combine output chunks
    Xi_parts = []
    ZZij_parts = []
    
    for chunk_file in chunk_files:
        data = np.load(chunk_file)
        Xi_parts.append(data['Xi'])
        ZZij_parts.append(data['ZZij'])
    
    Xi = np.vstack(Xi_parts)
    ZZij = np.vstack(ZZij_parts)
    
    del Xi_parts, ZZij_parts
    gc.collect()
    
    # Build final arrays
    A = np.hstack([all_h_i, all_J_ij, all_theta.reshape(-1, 1)])
    X = np.hstack([Xi, all_J_ij])
    
    # =========================================================================
    # Cleanup temp files
    # =========================================================================
    if verbose:
        print(f"  Cleaning up temp files...")
    
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"    Warning: Could not delete temp dir: {e}")
    
    return {"X": X, "y": ZZij, "A": A}


# =============================================================================
# Saving Data
# =============================================================================

def split_data(
    data: Dict[str, np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict, Dict, Dict]:
    """Split into train/val/test."""
    rng = np.random.default_rng(seed)
    n = len(data["X"])
    idx = rng.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    def subset(indices):
        return {k: v[indices] for k, v in data.items()}
    
    return (
        subset(idx[:n_train]),
        subset(idx[n_train:n_train + n_val]),
        subset(idx[n_train + n_val:]),
    )


def save_split(data: Dict[str, np.ndarray], path: Path):
    """Save as .pt file."""
    tensor_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
    torch.save(tensor_data, path)
    print(f"    {path.name}: {len(data['X'])} samples")


def save_metadata(output_dir: Path, args, shapes: Dict[str, tuple], config_name: str):
    """Save generation parameters as YAML."""
    import yaml
    
    n_edges = len(list(itertools.combinations(range(args.n_qubits), 2)))
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "config_file": f"configs/data/{config_name}.yaml",
        "distribution": args.dist,
        "n_samples": args.n_samples,
        "n_qubits": args.n_qubits,
        "n_edges": n_edges,
        "seed": args.seed,
        "splits": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": round(1 - args.train_ratio - args.val_ratio, 2),
        },
        "shapes": {k: list(v) for k, v in shapes.items()},
        "columns": {
            "X": f"[Xi ({args.n_qubits}), Jij ({n_edges})] -> input_dim={args.n_qubits + n_edges}",
            "y": f"[ZZij ({n_edges})] -> output_dim={n_edges}",
            "A": f"[hi ({args.n_qubits}), Jij ({n_edges}), theta (1)] -> algo_input",
            "Xi": f"[Xi ({args.n_qubits})] -> raw oracle output",
        },
    }
    
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"    metadata.yaml")


# =============================================================================
# Config Generation
# =============================================================================

def generate_tensor_config(
    config_name: str,
    data_dir: Path,
    n_qubits: int,
    n_edges: int,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> Path:
    """Generate Hydra config YAML for tensor (MLP) dataset."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config_name}.yaml"
    
    try:
        data_dir_rel = data_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        data_dir_rel = data_dir
    
    input_dim = n_qubits + n_edges
    output_dim = n_edges
    
    config_content = f'''# =============================================================================
# {config_name} Dataset Configuration (Tensor/MLP)
# =============================================================================
# Auto-generated by generate_data.py on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# Data dimensions:
#   input_dim:  {input_dim} = n_qubits ({n_qubits}) + n_edges ({n_edges})
#   output_dim: {output_dim} = n_edges ({n_edges})
#
# ML task: [Xi, Jij] -> [ZZij]
# =============================================================================

_target_: data.QuantumDataset
name: "{config_name}"

root: "{data_dir_rel}"
train_file: "train.pt"
val_file: "val.pt"
test_file: "test.pt"

is_graph: false

input_dim: {input_dim}
output_dim: {output_dim}
n_qubits: {n_qubits}
n_edges: {n_edges}

batch_size: ${{training.batch_size}}
num_workers: 4
pin_memory: true
'''
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    return config_path


def generate_graph_config(
    config_name: str,
    data_dir: Path,
    n_qubits: int,
    n_edges: int,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> Path:
    """Generate Hydra config YAML for graph (GNN) dataset."""
    config_dir.mkdir(parents=True, exist_ok=True)
    graph_config_name = f"{config_name}_graph"
    config_path = config_dir / f"{graph_config_name}.yaml"
    
    try:
        data_dir_rel = data_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        data_dir_rel = data_dir
    
    input_dim = n_qubits + n_edges
    output_dim = n_edges
    
    config_content = f'''# =============================================================================
# {graph_config_name} Dataset Configuration (Graph/GNN)
# =============================================================================
# Auto-generated by generate_data.py on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# Graph version for GNN training. Same data files as {config_name}.
#
# Data dimensions:
#   input_dim:  {input_dim} = n_qubits ({n_qubits}) + n_edges ({n_edges})
#   output_dim: {output_dim} = n_edges ({n_edges})
#
# ML task: [Xi, Jij] -> [ZZij]
# =============================================================================

_target_: data.QuantumGraphDataset
name: "{graph_config_name}"

root: "{data_dir_rel}"
train_file: "train.pt"
val_file: "val.pt"
test_file: "test.pt"

is_graph: true

# REQUIRED for graph dataset
n_qubits: {n_qubits}

input_dim: {input_dim}
output_dim: {output_dim}
n_edges: {n_edges}

batch_size: ${{training.batch_size}}
num_workers: 4
pin_memory: true
'''
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    return config_path


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate quantum ML dataset with auto-config creation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--dist", required=True, choices=["random", "sobol", "hybrid", "sweep", "gaussian"],
                        help="Distribution type (random, sobol, hybrid, sweep, gaussian)")
    parser.add_argument("--n_samples", type=int, required=True,
                        help="Number of samples (for sweep: must equal n_base × n_theta)")
    parser.add_argument("--n_qubits", type=int, required=True,
                        help="Number of qubits")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/{dist}_{n}q)")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Config name (default: {dist}_{n}q)")
    parser.add_argument("--julia_file", type=str, default=None,
                        help="Julia file path")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training split")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Generate in chunks of this size (helps with memory for large datasets, e.g., 100000)")
    
    # Hybrid distribution options
    parser.add_argument("--sobol_fraction", type=float, default=0.5,
                        help="Fraction of samples from Sobol (for hybrid dist, default 0.5)")
    parser.add_argument("--boundary_width", type=float, default=0.1,
                        help="Boundary region width in [0,1] space (for hybrid dist, default 0.1)")
    
    # Sweep distribution options
    parser.add_argument("--sweep_n_base", type=int, default=None,
                        help="Number of base [h_i, J_ij] configs (for sweep dist). "
                             "If not set, computed as n_samples / sweep_n_theta")
    parser.add_argument("--sweep_n_theta", type=int, default=10,
                        help="Number of theta values to sweep (for sweep dist, default 10)")
    parser.add_argument("--sweep_base_dist", choices=["random", "sobol"], default="random",
                        help="Distribution for base [h_i, J_ij] samples (for sweep dist)")
    
    parser.add_argument("--no_config", action="store_true", help="Skip config generation")
    parser.add_argument("--graph_config", action="store_true", help="Also generate graph config")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress")
    
    # Safety flags
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing data (DANGEROUS)")
    parser.add_argument("--append_timestamp", action="store_true",
                        help="Append timestamp to output directory name")
    
    return parser.parse_args()


def check_existing_data(output_dir: Path, overwrite: bool) -> bool:
    """
    Check if data already exists and handle accordingly.
    
    Returns True if safe to proceed, False if should abort.
    """
    if not output_dir.exists():
        return True
    
    # Check for existing .pt files
    existing_files = list(output_dir.glob("*.pt"))
    
    if not existing_files:
        return True
    
    print()
    print("!" * 60)
    print("WARNING: Data already exists in this directory!")
    print("!" * 60)
    print(f"  Directory: {output_dir}")
    print(f"  Existing files:")
    for f in existing_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.name} ({size_mb:.1f} MB)")
    print()
    
    if overwrite:
        print("--overwrite flag set. Existing data will be DELETED.")
        response = input("Type 'YES' to confirm deletion: ")
        if response != "YES":
            print("Aborted.")
            return False
        
        # Delete existing files
        for f in existing_files:
            f.unlink()
            print(f"  Deleted: {f.name}")
        
        # Also delete metadata
        metadata_file = output_dir / "metadata.yaml"
        if metadata_file.exists():
            metadata_file.unlink()
            print(f"  Deleted: metadata.yaml")
        
        print()
        return True
    
    else:
        print("To overwrite, use --overwrite flag (requires confirmation)")
        print("To create new dataset, use --output or --append_timestamp")
        print()
        print("Suggestions:")
        print(f"  1. Different name:  --output data/{output_dir.name}_v2")
        print(f"  2. With timestamp:  --append_timestamp")
        print(f"  3. Force overwrite: --overwrite")
        print()
        return False


def main():
    args = parse_args()
    
    n_edges = len(list(itertools.combinations(range(args.n_qubits), 2)))
    config_name = args.config_name or f"{args.dist}_{args.n_qubits}q"
    
    julia_file = Path(args.julia_file) if args.julia_file else DEFAULT_JULIA_FILE
    
    # Handle sweep distribution parameters
    sweep_n_base = args.sweep_n_base
    if args.dist == "sweep":
        if sweep_n_base is None:
            # Compute n_base from n_samples and n_theta
            if args.n_samples % args.sweep_n_theta != 0:
                print(f"ERROR: For sweep distribution, n_samples ({args.n_samples}) must be "
                      f"divisible by sweep_n_theta ({args.sweep_n_theta})")
                print(f"  Try: --n_samples {(args.n_samples // args.sweep_n_theta) * args.sweep_n_theta}")
                return
            sweep_n_base = args.n_samples // args.sweep_n_theta
        else:
            # Validate consistency
            expected_samples = sweep_n_base * args.sweep_n_theta
            if expected_samples != args.n_samples:
                print(f"ERROR: n_samples ({args.n_samples}) != sweep_n_base ({sweep_n_base}) × "
                      f"sweep_n_theta ({args.sweep_n_theta}) = {expected_samples}")
                return
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        base_name = config_name
        if args.append_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{config_name}_{timestamp}"
            config_name = base_name  # Update config name too
        output_dir = DEFAULT_DATA_DIR / base_name
    
    # Safety check for existing data
    if not check_existing_data(output_dir, args.overwrite):
        print("Aborted. No data was generated.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Quantum Data Generation")
    print("=" * 60)
    print(f"  Distribution:  {args.dist}")
    if args.dist == "hybrid":
        print(f"    Sobol frac:  {args.sobol_fraction} ({int(args.n_samples * args.sobol_fraction):,} samples)")
        print(f"    Stratified:  {1 - args.sobol_fraction} ({args.n_samples - int(args.n_samples * args.sobol_fraction):,} samples)")
        print(f"    Boundary:    {args.boundary_width} (J in [0,{args.boundary_width}] or [{1-args.boundary_width},1])")
    elif args.dist == "sweep":
        print(f"    Base configs: {sweep_n_base:,} [h_i, J_ij] points ({args.sweep_base_dist})")
        print(f"    Theta sweep:  {args.sweep_n_theta} points in [0, π/2]")
        print(f"    Total:        {sweep_n_base:,} × {args.sweep_n_theta} = {args.n_samples:,} samples")
    print(f"  Samples:       {args.n_samples:,}")
    print(f"  Qubits:        {args.n_qubits}")
    print(f"  Edges:         {n_edges}")
    print(f"  Seed:          {args.seed}")
    print(f"  Chunk size:    {args.chunk_size or 'None (single batch)'}")
    print(f"  Output:        {output_dir}")
    print(f"  Config name:   {config_name}")
    print()
    
    print("[1/4] Initializing Julia...")
    julia_main = init_julia(julia_file)
    print("  Ready")
    print()
    
    print(f"[2/4] Generating {args.n_samples:,} samples...")
    data = generate_data(
        n_samples=args.n_samples,
        n_qubits=args.n_qubits,
        distribution=args.dist,
        julia_main=julia_main,
        seed=args.seed,
        verbose=not args.quiet,
        chunk_size=args.chunk_size,
        sobol_fraction=args.sobol_fraction,
        boundary_width=args.boundary_width,
        sweep_n_base=sweep_n_base,
        sweep_n_theta=args.sweep_n_theta,
        sweep_base_dist=args.sweep_base_dist,
    )
    
    print("  Shapes:")
    for k, v in data.items():
        print(f"    {k}: {v.shape}")
    print()
    
    print("[3/4] Splitting and saving data...")
    train, val, test = split_data(data, args.train_ratio, args.val_ratio, args.seed)
    
    print(f"  Saving to {output_dir}/")
    save_split(train, output_dir / "train.pt")
    save_split(val, output_dir / "val.pt")
    save_split(test, output_dir / "test.pt")
    save_metadata(output_dir, args, {k: v.shape for k, v in data.items()}, config_name)
    print()
    
    print("[4/4] Generating config file(s)...")
    
    if not args.no_config:
        config_path = generate_tensor_config(
            config_name=config_name,
            data_dir=output_dir,
            n_qubits=args.n_qubits,
            n_edges=n_edges,
        )
        print(f"  Created: {config_path.relative_to(PROJECT_ROOT)}")
        
        # Always create graph config too (useful for both MLP and GNN)
        graph_config_path = generate_graph_config(
            config_name=config_name,
            data_dir=output_dir,
            n_qubits=args.n_qubits,
            n_edges=n_edges,
        )
        print(f"  Created: {graph_config_path.relative_to(PROJECT_ROOT)}")
    else:
        print("  Skipped (--no_config)")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print()
    print("Train with:")
    print(f"  python src/train.py data={config_name} model=mlp")
    print(f"  python src/train.py data={config_name}_graph model=gnn")
    print()
    print("Quick test:")
    print(f"  python src/train.py experiment=debug data={config_name}")


if __name__ == "__main__":
    main()
