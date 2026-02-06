#!/usr/bin/env python3
"""
Data Generation Script - Generates vdat data using Julia oracle.

Usage:
    python scripts/generate_data.py --dist random --n_samples 1000 --n_qubits 4
"""

import argparse
import itertools
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
from string import Template

import numpy as np
import torch
import yaml
import time
from scipy.stats import qmc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_JULIA_FILE = PROJECT_ROOT / "src" / "gene_transverse_field_Ising.jl"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "data"
TEMPLATE_DIR = PROJECT_ROOT / "configs" / "templates"


def init_julia(hamiltonian_path: Path):
    print(f"  Julia file: {hamiltonian_path}")
    if not hamiltonian_path.exists():
        raise FileNotFoundError(f"Julia file not found: {hamiltonian_path}")
    
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
    Xi_raw, ZZij_raw = Main.oracle(Jij_dict, h_i, theta, n_qubits, edges)
    Xi = np.ravel(np.asarray(Xi_raw, dtype=np.float64))
    ZZij = np.ravel(np.asarray(ZZij_raw, dtype=np.float64))
    return Xi, ZZij


def sample_random(n_qubits: int, n_edges: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    h_i = rng.uniform(0.0, 1.0, size=n_qubits)
    J_ij = rng.uniform(-1.0, 1.0, size=n_edges)
    theta = rng.uniform(0.0, np.pi / 2)
    return h_i, J_ij, theta


def sample_sobol(n_qubits: int, n_edges: int, sobol_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    idx = 0
    h_i = sobol_sample[idx:idx + n_qubits]
    idx += n_qubits
    J_ij = sobol_sample[idx:idx + n_edges] * 2 - 1  # [0,1] -> [-1,1]
    idx += n_edges
    theta = sobol_sample[idx] * (np.pi / 2)
    return h_i, J_ij, theta


def sample_ising(n_qubits: int, n_edges: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    h_i = np.zeros(n_qubits)
    J_ij = rng.uniform(-0.5,0.5,size=n_edges)
    J_ij = J_ij / np.linalg.norm(J_ij)
    theta = np.pi / 2
    return h_i, J_ij, theta

def sample_normal(n_qubits: int, n_edges: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    h_i = rng.uniform(0,1,size=n_qubits)
    J_ij = rng.uniform(-0.5,0.5,size=n_edges)
    J_ij = J_ij / np.linalg.norm(J_ij)
    theta = None # sweep 10 thetas per config
    return h_i, J_ij, theta

def sample_coulomb(n_qubits: int, n_edges: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    # positive version of sample_ising
    h_i = np.zeros(n_qubits)
    J_ij = rng.uniform(0,0.5,size=n_edges)
    J_ij = J_ij / np.linalg.norm(J_ij)
    theta = np.pi / 2
    return h_i, J_ij, theta

def sample_uniform_j(n_qubits: int, n_edges: int, rng: np.random.Generator):
    J_ij = np.full(n_edges, 1.0 / np.sqrt(n_edges))  # all equal, ||J||=1
    h_total = rng.uniform(0.0, 1.0)
    h_i = np.full(n_qubits, h_total / n_qubits)  # uniform h, sum = h_total
    return h_i, J_ij, np.linspace(0, np.pi / 2, 10)

def _hypersphere(n, rng):
    J = rng.standard_normal(n)
    return J / np.linalg.norm(J)

def sample_dimer(n_qubits: int, n_edges: int, rng: np.random.Generator):
    h_i = rng.uniform(0, 1, size=n_qubits)
    J_ij = np.zeros(n_edges)
    J_ij[rng.integers(0, n_edges)] = rng.choice([-1.0, 1.0])
    return h_i, J_ij, np.linspace(0, np.pi / 2, 10)

def sample_near_ising(n_qubits: int, n_edges: int, rng: np.random.Generator):
    h_i = rng.uniform(0, 1, size=n_qubits)
    J_ij = _hypersphere(n_edges, rng)
    return h_i, J_ij, np.linspace(np.pi/2 - 0.3, np.pi/2, 10)

def sample_field(n_qubits: int, n_edges: int, rng: np.random.Generator):
    h_i = rng.uniform(0, 1, size=n_qubits)
    J_ij = _hypersphere(n_edges, rng)
    return h_i, J_ij, np.linspace(0, 0.3, 10)

MIXED_CASES = [
    (sample_normal,     0.333),  # 30%
    (sample_ising,      0.222),  # 20%
    (sample_dimer,      0.111),  # 10%
    (sample_near_ising, 0.167),  # 15%
    (sample_field,      0.167),  # 15%
]
_MIXED_SAMPLERS = [c[0] for c in MIXED_CASES]
_MIXED_WEIGHTS = np.array([c[1] for c in MIXED_CASES])
_MIXED_WEIGHTS /= _MIXED_WEIGHTS.sum()

def sample_mixed(n_qubits: int, n_edges: int, rng: np.random.Generator):
    idx = rng.choice(len(_MIXED_SAMPLERS), p=_MIXED_WEIGHTS)
    return _MIXED_SAMPLERS[idx](n_qubits, n_edges, rng)

DISTRIBUTIONS = {
    "random": sample_random, "sobol": sample_sobol, "normal": sample_normal,
    "coulomb": sample_coulomb, "ising": sample_ising,
    "dimer": sample_dimer, "near_ising": sample_near_ising, "field": sample_field,
    "mixed": sample_mixed,
    "uniform_j": sample_uniform_j,
}

def generate_data(
    n_samples: int,
    n_qubits: int,
    distribution: str,
    julia_main,
    seed: int = 42,
    verbose: bool = True,
    batch_size: int = 10000,          # flush every N configs to avoid Julia GC segfault
) -> Dict[str, np.ndarray]:
    
    rng = np.random.default_rng(seed)
    edges_julia = list(itertools.combinations(range(1, n_qubits + 1), 2))
    n_edges = len(edges_julia)
    
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    sobol_samples = None
    if distribution == "sobol":
        dim = n_qubits + n_edges + 1
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        sobol_samples = sampler.random(n_samples)
    
    print_freq = max(1, n_samples // 10)
    
    # Batched storage — flush periodically to reduce memory pressure
    A_batch, Xi_batch, ZZij_batch = [], [], []
    A_all, Xi_all, ZZij_all = [], [], []
    oracle_count = 0
    start_time = time.time()
    
    for i in range(n_samples):
        if verbose and (i + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            rate = oracle_count / elapsed if elapsed > 0 else 0
            pct = 100 * (i + 1) / n_samples
            print(f"    {i + 1:>8,} / {n_samples:,} ({pct:5.1f}%) | Oracle calls: {oracle_count:,} | Rate: {rate:.1f}/s")
       
        if distribution == "sobol":
            h_i, J_ij, theta = sample_sobol(n_qubits, n_edges, sobol_samples[i])
        else:
            h_i, J_ij, theta = DISTRIBUTIONS[distribution](n_qubits, n_edges, rng)
        
        Jij_dict = {edge: J_ij[k] for k, edge in enumerate(edges_julia)}

        if theta is None:
            thetas = np.linspace(0, np.pi/2, 10)
        elif isinstance(theta, np.ndarray):
            thetas = theta
        else:
            thetas = [theta]

        for t in thetas:
            A_batch.append(np.concatenate([h_i, J_ij, [t]]))
            Xi, ZZij = call_oracle(julia_main, Jij_dict, h_i, t, n_qubits, edges_julia)
            Xi_batch.append(Xi)
            ZZij_batch.append(ZZij)
            oracle_count += 1

        # Flush batch: vstack into numpy arrays, clear lists, nudge Julia GC
        if (i + 1) % batch_size == 0:
            A_all.append(np.vstack(A_batch))
            Xi_all.append(np.vstack(Xi_batch))
            ZZij_all.append(np.vstack(ZZij_batch))
            A_batch, Xi_batch, ZZij_batch = [], [], []
            try:
                julia_main.eval("GC.gc()")
            except Exception:
                pass
    
    # Flush remaining
    if A_batch:
        A_all.append(np.vstack(A_batch))
        Xi_all.append(np.vstack(Xi_batch))
        ZZij_all.append(np.vstack(ZZij_batch))
    
    if verbose:
        total = time.time() - start_time
        print(f"\n  Done: {oracle_count:,} oracle calls from {n_samples:,} configs in {total:.1f}s ({oracle_count/total:.1f}/s)")
    
    A = np.vstack(A_all)
    Xi = np.vstack(Xi_all)
    ZZij = np.vstack(ZZij_all)
    Jij = A[:, n_qubits:n_qubits + n_edges]
    X = np.hstack([Xi, Jij])
    
    return {"X": X, "y": ZZij, "A": A, "Xi": Xi}


def split_data(data: Dict[str, np.ndarray], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(data["X"])
    idx = rng.permutation(n)
    n_train, n_val = int(n * train_ratio), int(n * val_ratio)
    
    def subset(indices):
        return {k: v[indices] for k, v in data.items()}
    
    return subset(idx[:n_train]), subset(idx[n_train:n_train + n_val]), subset(idx[n_train + n_val:])


def save_split(data: Dict[str, np.ndarray], path: Path):
    torch.save({k: torch.tensor(v, dtype=torch.float64) for k, v in data.items()}, path)
    print(f"    {path.name}: {len(data['X'])} samples")


def save_metadata(output_dir: Path, args, shapes: Dict[str, tuple], config_name: str):
    n_edges = len(list(itertools.combinations(range(args.n_qubits), 2)))
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "config_file": f"configs/data/{config_name}.yaml",
        "distribution": args.dist,
        "n_samples": args.n_samples,
        "n_qubits": args.n_qubits,
        "n_edges": n_edges,
        "seed": args.seed,
        "splits": {"train": args.train_ratio, "val": args.val_ratio, "test": round(1 - args.train_ratio - args.val_ratio, 2)},
        "shapes": {k: list(v) for k, v in shapes.items()},
    }
    
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"    metadata.yaml")


def generate_config(config_name: str, data_dir: Path, n_qubits: int, n_edges: int, is_graph: bool = False):
    template_file = TEMPLATE_DIR / ("graph_config.yaml" if is_graph else "tensor_config.yaml")
    output_name = f"{config_name}_graph.yaml" if is_graph else f"{config_name}.yaml"
    output_path = DEFAULT_CONFIG_DIR / output_name
    
    try:
        data_dir_rel = data_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        data_dir_rel = data_dir
    
    with open(template_file) as f:
        template = Template(f.read())
    
    content = template.substitute(
        config_name=config_name + ("_graph" if is_graph else ""),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_dir=data_dir_rel,
        input_dim=n_qubits + n_edges,
        output_dim=n_edges,
        n_qubits=n_qubits,
        n_edges=n_edges,
    )
    
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VDAT ML dataset")
    parser.add_argument("--dist", required=True, choices=list(DISTRIBUTIONS.keys()))
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--n_qubits", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--julia_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--no_config", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    n_edges = len(list(itertools.combinations(range(args.n_qubits), 2)))
    config_name = args.config_name or f"{args.dist}_{args.n_qubits}q"
    julia_file = Path(args.julia_file) if args.julia_file else DEFAULT_JULIA_FILE
    output_dir = Path(args.output) if args.output else DEFAULT_DATA_DIR / config_name
    
    if output_dir.exists() and list(output_dir.glob("*.pt")) and not args.overwrite:
        print(f"Data exists at {output_dir}. Use --overwrite to replace.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print(f"  Distribution: {args.dist} | Samples: {args.n_samples} | Qubits: {args.n_qubits}")
    print("=" * 50)
    
    print("\n[1/4] Initializing Julia...")
    julia_main = init_julia(julia_file)
    
    print(f"\n[2/4] Generating {args.n_samples} samples...")
    data = generate_data(args.n_samples, args.n_qubits, args.dist, julia_main, args.seed, not args.quiet)
    
    print("\n[3/4] Saving data...")
    train, val, test = split_data(data, args.train_ratio, args.val_ratio, args.seed)
    save_split(train, output_dir / "train.pt")
    save_split(val, output_dir / "val.pt")
    save_split(test, output_dir / "test.pt")
    save_metadata(output_dir, args, {k: v.shape for k, v in data.items()}, config_name)
    
    print("\n[4/4] Generating configs...")
    if not args.no_config:
        for is_graph in [False, True]:
            path = generate_config(config_name, output_dir, args.n_qubits, n_edges, is_graph)
            print(f"    {path.relative_to(PROJECT_ROOT)}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
