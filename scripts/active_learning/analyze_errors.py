#!/usr/bin/env python3
"""
Error Analysis and Error Predictor Training
============================================

Given a trained GNN checkpoint and dataset:
1. Evaluate GNN on validation/test set
2. Compute per-sample errors
3. Analyze correlation between error and Hamiltonian parameters [h_i, J_ij, θ]
4. Train an error predictor: [h_i, J_ij, θ] → expected_error
5. Save error predictor for use in active learning

The error predictor can then be used to score AL candidates by predicted error
(targeting where the model is WRONG, not just UNCERTAIN).

Usage:
    python scripts/active_learning/analyze_errors.py \
        --checkpoint outputs/gnn_run/best_model.pt \
        --data_dir data/random_4q \
        --output models/error_predictor.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import GNN
from data import QuantumGraphDataset


class ErrorPredictor(nn.Module):
    """
    Simple MLP that predicts GNN error from Hamiltonian parameters.
    
    Input: [h_i, J_ij, θ] (n_qubits + n_edges + 1 dims)
    Output: predicted error (scalar, mean absolute error across ZZ_ij)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
            ])
            prev_dim = h_dim
        
        # Output: positive error (use softplus to ensure positive)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_gnn_checkpoint(checkpoint_path: Path, device: str = 'cpu'):
    """Load GNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        model = GNN(
            node_dim=cfg.get('node_dim', 1),
            edge_dim=cfg.get('edge_dim', 1),
            hidden_dim=cfg.get('hidden_dim', 128),
            num_layers=cfg.get('num_layers', 4),
            activation=cfg.get('activation', 'silu'),
            dropout=cfg.get('dropout', 0.0),
        )
    else:
        # Default config if not saved
        print("  Warning: No config in checkpoint, using defaults")
        model = GNN(hidden_dim=128, num_layers=4)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def load_data_with_A(data_dir: Path, split: str = 'val'):
    """Load data including Hamiltonian parameters A."""
    data_path = data_dir / f"{split}.pt"
    data = torch.load(data_path, weights_only=False)
    
    X = data['X']  # [X_i, J_ij]
    y = data['y']  # ZZ_ij
    A = data.get('A', None)  # [h_i, J_ij, θ]
    
    if A is None:
        raise ValueError(
            f"Data file {data_path} does not contain 'A' (Hamiltonian parameters).\n"
            "Regenerate data with a recent version of generate_data.py"
        )
    
    return X, y, A


def compute_errors(model, X, y, n_qubits: int, batch_size: int = 256, device: str = 'cpu'):
    """
    Compute GNN prediction errors on dataset.
    
    Returns per-sample MAE (mean across ZZ_ij outputs).
    """
    from torch_geometric.data import Data, Batch
    
    n_samples = len(X)
    all_errors = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Convert to PyG Data objects
            data_list = []
            for j in range(len(batch_X)):
                x_j = batch_X[j]
                y_j = batch_y[j]
                
                # Extract node and edge features
                Xi = x_j[:n_qubits].unsqueeze(-1)  # (n_qubits, 1)
                J_ij = x_j[n_qubits:]  # (n_edges,)
                
                # Build fully connected edge index (upper triangle)
                edge_index = []
                edge_attr = []
                idx = 0
                for src in range(n_qubits):
                    for dst in range(src + 1, n_qubits):
                        edge_index.append([src, dst])
                        edge_attr.append(J_ij[idx].item())
                        idx += 1
                
                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
                
                data = Data(x=Xi, edge_index=edge_index, edge_attr=edge_attr, y=y_j)
                data_list.append(data)
            
            batch = Batch.from_data_list(data_list).to(device)
            
            # Get predictions
            pred = model(batch)
            
            # Reshape to (batch_size, n_edges)
            n_edges = y.size(1)
            pred = pred.view(-1, n_edges)
            target = batch_y.to(device)
            
            # Per-sample MAE
            errors = (pred - target).abs().mean(dim=-1)
            all_errors.append(errors.cpu())
    
    return torch.cat(all_errors)


def analyze_error_patterns(errors, A, n_qubits: int):
    """Analyze how errors correlate with Hamiltonian parameters."""
    n_edges = A.size(1) - n_qubits - 1  # A = [h_i, J_ij, θ]
    
    J_ij = A[:, n_qubits:n_qubits + n_edges]
    theta = A[:, -1]
    
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    # Basic stats
    print(f"\nError statistics:")
    print(f"  Mean:   {errors.mean():.6f}")
    print(f"  Std:    {errors.std():.6f}")
    print(f"  Median: {errors.median():.6f}")
    print(f"  Max:    {errors.max():.6f}")
    print(f"  Min:    {errors.min():.6f}")
    
    # Top 1% errors
    k = max(1, len(errors) // 100)
    top_errors, top_idx = errors.topk(k)
    print(f"\nTop {k} errors (1%): {top_errors.mean():.6f} mean")
    
    # J_ij patterns in high-error samples
    top_J = J_ij[top_idx]
    all_J = J_ij
    
    # Extreme J analysis
    thresholds = [0.8, 0.9, 0.95]
    print(f"\nFraction of |J_ij| > threshold:")
    print(f"  {'Threshold':<12} {'All samples':<15} {'High-error (top 1%)':<20} {'Ratio':<10}")
    print(f"  {'-'*57}")
    
    for thresh in thresholds:
        all_extreme = (all_J.abs() > thresh).float().mean().item()
        top_extreme = (top_J.abs() > thresh).float().mean().item()
        ratio = top_extreme / max(all_extreme, 1e-6)
        print(f"  {thresh:<12} {all_extreme:<15.3f} {top_extreme:<20.3f} {ratio:<10.1f}x")
    
    # Number of extreme dimensions
    print(f"\nNumber of extreme dims (|J| > 0.9) per sample:")
    all_n_extreme = (all_J.abs() > 0.9).sum(dim=-1).float()
    top_n_extreme = (top_J.abs() > 0.9).sum(dim=-1).float()
    print(f"  All samples:    {all_n_extreme.mean():.2f} ± {all_n_extreme.std():.2f}")
    print(f"  High-error:     {top_n_extreme.mean():.2f} ± {top_n_extreme.std():.2f}")
    
    # Theta analysis
    print(f"\nTheta distribution:")
    all_theta = theta
    top_theta = theta[top_idx]
    print(f"  All samples:    {all_theta.mean():.3f} ± {all_theta.std():.3f}")
    print(f"  High-error:     {top_theta.mean():.3f} ± {top_theta.std():.3f}")
    
    # Error by |J| bins
    print(f"\nMean error by mean |J_ij|:")
    mean_abs_J = all_J.abs().mean(dim=-1)
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    for i in range(len(bins) - 1):
        mask = (mean_abs_J >= bins[i]) & (mean_abs_J < bins[i+1])
        if mask.sum() > 0:
            bin_error = errors[mask].mean()
            print(f"  |J| ∈ [{bins[i]:.2f}, {bins[i+1]:.2f}): {bin_error:.6f} (n={mask.sum():,})")
    
    print("=" * 60)


def train_error_predictor(A, errors, n_epochs: int = 100, batch_size: int = 256, 
                          lr: float = 1e-3, device: str = 'cpu', verbose: bool = True):
    """Train error predictor on validation data."""
    input_dim = A.size(1)
    
    model = ErrorPredictor(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    # Use log-transformed error for better training (errors are positive, often small)
    log_errors = torch.log1p(errors * 100)  # log(1 + 100*error) for numerical stability
    
    dataset = TensorDataset(A, log_errors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for A_batch, err_batch in loader:
            A_batch = A_batch.to(device)
            err_batch = err_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(A_batch)
            
            # MSE on log-transformed error
            loss = ((pred - err_batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {epoch_loss/len(loader):.6f}")
    
    model.eval()
    return model


def evaluate_predictor(model, A, errors, device: str = 'cpu'):
    """Evaluate error predictor quality."""
    model.eval()
    
    log_errors = torch.log1p(errors * 100)
    
    with torch.no_grad():
        pred = model(A.to(device)).cpu()
    
    # Correlation between predicted and actual
    pred_np = pred.numpy()
    actual_np = log_errors.numpy()
    correlation = np.corrcoef(pred_np, actual_np)[0, 1]
    
    # Does it rank high-error samples correctly?
    k = len(errors) // 100  # top 1%
    _, true_top_idx = errors.topk(k)
    _, pred_top_idx = pred.topk(k)
    
    # Overlap between predicted top-1% and actual top-1%
    true_set = set(true_top_idx.numpy())
    pred_set = set(pred_top_idx.numpy())
    overlap = len(true_set & pred_set) / k
    
    print(f"\nError predictor evaluation:")
    print(f"  Correlation (log error): {correlation:.3f}")
    print(f"  Top-1% overlap:          {overlap:.1%}")
    
    return correlation, overlap


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GNN errors and train error predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--checkpoint", required=True, help="Path to GNN checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--output", default="models/error_predictor.pt",
                        help="Output path for error predictor")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Data split to analyze")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--epochs", type=int, default=100, help="Error predictor training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    print("=" * 60)
    print("GNN ERROR ANALYSIS & PREDICTOR TRAINING")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data:       {data_dir}")
    print(f"  Split:      {args.split}")
    print(f"  Output:     {output_path}")
    print(f"  Device:     {args.device}")
    print()
    
    # Load model
    print("[1/5] Loading GNN checkpoint...")
    model = load_gnn_checkpoint(checkpoint_path, args.device)
    print(f"  Loaded")
    
    # Load data
    print(f"\n[2/5] Loading {args.split} data...")
    X, y, A = load_data_with_A(data_dir, args.split)
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  A: {A.shape}")
    
    # Compute errors
    print(f"\n[3/5] Computing GNN errors...")
    errors = compute_errors(model, X, y, args.n_qubits, args.batch_size, args.device)
    print(f"  Computed {len(errors):,} errors")
    
    # Analyze patterns
    print(f"\n[4/5] Analyzing error patterns...")
    analyze_error_patterns(errors, A, args.n_qubits)
    
    # Train error predictor
    print(f"\n[5/5] Training error predictor...")
    error_predictor = train_error_predictor(
        A, errors, 
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Evaluate
    evaluate_predictor(error_predictor, A, errors, args.device)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': error_predictor.state_dict(),
        'input_dim': A.size(1),
        'n_qubits': args.n_qubits,
        'source_checkpoint': str(checkpoint_path),
        'source_data': str(data_dir),
    }, output_path)
    print(f"\n  Saved error predictor to: {output_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nUse in active learning:")
    print(f"  python scripts/active_learning/run_al.py \\")
    print(f"      --surrogate error_predictor \\")
    print(f"      --error_predictor_path {output_path} \\")
    print(f"      ...")


if __name__ == "__main__":
    main()
