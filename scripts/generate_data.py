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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """TODO: Implement truncated Gaussian sampling."""
    raise NotImplementedError(
        "Gaussian distribution not yet implemented.\n"
        "Add your truncated normal logic here."
    )


DISTRIBUTIONS = {
    "random": sample_random,
    "gaussian": sample_gaussian,
}


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
) -> Dict[str, np.ndarray]:
    """
    Generate quantum ML dataset.
    
    Args:
        n_samples: Total number of samples to generate
        n_qubits: Number of qubits
        distribution: Distribution name
        julia_main: Initialized Julia Main module
        seed: Random seed
        verbose: Print progress
        chunk_size: If set, generate in chunks and GC between them
    
    Returns dict with:
        X:  ML input [Xi, Jij], shape (N, n_qubits + n_edges)
        y:  ML target [ZZij], shape (N, n_edges)
        A:  Algorithm input [hi, Jij, theta], shape (N, n_qubits + n_edges + 1)
        Xi: Raw Xi, shape (N, n_qubits)
    """
    import gc
    import time
    
    rng = np.random.default_rng(seed)
    # Julia uses 1-indexed edges
    edges_julia = list(itertools.combinations(range(1, n_qubits + 1), 2))
    n_edges = len(edges_julia)
    
    sampler = DISTRIBUTIONS.get(distribution)
    if sampler is None:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = n_samples  # No chunking
    
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
    
    A_list, Xi_list, ZZij_list = [], [], []
    
    samples_generated = 0
    chunk_num = 0
    total_start_time = time.time()
    
    while samples_generated < n_samples:
        chunk_num += 1
        chunk_start = samples_generated
        chunk_end = min(samples_generated + chunk_size, n_samples)
        chunk_samples = chunk_end - chunk_start
        
        if verbose and chunk_size < n_samples:
            print(f"\n  [Chunk {chunk_num}] Generating samples {chunk_start + 1:,} to {chunk_end:,}")
        
        chunk_start_time = time.time()
        
        for i in range(chunk_samples):
            global_idx = samples_generated + i
            
            if verbose and (global_idx + 1) % print_freq == 0:
                elapsed = time.time() - total_start_time
                rate = (global_idx + 1) / elapsed
                eta = (n_samples - global_idx - 1) / rate if rate > 0 else 0
                pct = 100 * (global_idx + 1) / n_samples
                print(f"    {global_idx + 1:>8,} / {n_samples:,} ({pct:5.1f}%) | "
                      f"Rate: {rate:.1f} samples/sec | ETA: {eta/60:.1f} min")
            
            h_i, J_ij, theta = sampler(n_qubits, n_edges, rng)
            
            A_i = np.concatenate([h_i, J_ij, [theta]])
            A_list.append(A_i)
            
            Jij_dict = {edge: J_ij[k] for k, edge in enumerate(edges_julia)}
            Xi, ZZij = call_oracle(julia_main, Jij_dict, h_i, theta, n_qubits, edges_julia)
            
            Xi_list.append(Xi)
            ZZij_list.append(ZZij)
        
        samples_generated = chunk_end
        
        # Chunk summary
        if verbose and chunk_size < n_samples:
            chunk_elapsed = time.time() - chunk_start_time
            print(f"    Chunk {chunk_num} complete: {chunk_samples:,} samples in {chunk_elapsed:.1f}s")
        
        # Force garbage collection between chunks
        if chunk_size < n_samples and samples_generated < n_samples:
            if verbose:
                print(f"    Running garbage collection...")
            gc.collect()
    
    # Final summary
    if verbose:
        total_elapsed = time.time() - total_start_time
        print(f"\n  Generation complete: {n_samples:,} samples in {total_elapsed:.1f}s ({n_samples/total_elapsed:.1f} samples/sec)")
    
    # Stack arrays
    A = np.vstack(A_list)
    Xi = np.vstack(Xi_list)
    ZZij = np.vstack(ZZij_list)
    
    # ML input: [Xi, Jij]
    Jij = A[:, n_qubits:n_qubits + n_edges]
    X = np.hstack([Xi, Jij])
    
    return {"X": X, "y": ZZij, "A": A, "Xi": Xi}


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
    
    parser.add_argument("--dist", required=True, choices=list(DISTRIBUTIONS.keys()),
                        help="Distribution type")
    parser.add_argument("--n_samples", type=int, required=True,
                        help="Number of samples")
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
    
    parser.add_argument("--no_config", action="store_true", help="Skip config generation")
    parser.add_argument("--graph_config", action="store_true", help="Also generate graph config")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress")
    
    # Safety flags
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing data (CAUTION)")
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
    print(f"  Samples:       {args.n_samples}")
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
    
    print(f"[2/4] Generating {args.n_samples} samples...")
    data = generate_data(
        n_samples=args.n_samples,
        n_qubits=args.n_qubits,
        distribution=args.dist,
        julia_main=julia_main,
        seed=args.seed,
        verbose=not args.quiet,
        chunk_size=args.chunk_size,
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

if __name__ == "__main__":
    main()
