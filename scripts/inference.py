#!/usr/bin/env python3
"""
Inference script for VDAT-ML.

Loads a trained model checkpoint, reads a .dat file from VDAT,
and plots O_truth vs O_pred for multiple r values.

Usage:
    python scripts/inference.py checkpoints/best.pt data/
    python scripts/inference.py checkpoints/best.pt data/ --r 0.25
    python scripts/inference.py checkpoints/best.pt data/ --save-only
"""

import argparse
import sys
import math
import itertools
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models import GNN, MLP
from torch import nn

# =============================================================================
# Data Loading
# =============================================================================

def load_dat_file(filepath: str, n_qubits: int = 4):
    """
    Load VDAT output .dat file.
    
    Format: θ, ξ_avg, O_truth, ξ_i[0], ξ_i[1], ..., ξ_i[n_qubits-1]
    
    Returns:
        theta: (N,) array
        xi_avg: (N,) array  
        O_truth: (N,) array
        xi_i: (N, n_qubits) array
    """
    data = np.loadtxt(filepath)
    
    theta = data[:, 0]
    xi_avg = data[:, 1]
    O_truth = data[:, 2]
    xi_i = data[:, 3:3 + n_qubits]
    
    return theta, xi_avg, O_truth, xi_i


def compute_Jij(r: float, n_qubits: int = 4):
    """
    Compute Jij coupling values for given r parameter.
    
    For 4 qubits:
        J12, J34 = 1.0, 1.0
        J14, J23 = (1 - 2*r)
        J13, J24 = (1 - 3*r)
    
    Returns:
        Jij: (n_edges,) array in canonical edge order
    """
    if n_qubits != 4: # currently im only doing 4 qubits
        raise NotImplementedError("Only 4 qubits supported currently")
    
    J12, J34 = 1.0, 1.0
    J14, J23 = (1 - 2*r), (1 - 2*r)
    J13, J24 = (1 - 3*r), (1 - 3*r)
    
    return np.array([J12, J13, J14, J23, J24, J34])


def create_graph_dataset(Xi, Jij, edge_index):
    """
    Create PyG Data objects from Xi and Jij arrays.
    
    Args:
        Xi: (N, n_qubits) node features
        Jij: (N, n_edges) edge features
        edge_index: (2, n_edges) edge connectivity
    
    Returns:
        List of PyG Data objects
    """
    data_list = []
    num_samples = Xi.shape[0]
    
    for i in range(num_samples):
        x_tensor = torch.tensor(Xi[i], dtype=torch.float).unsqueeze(1)
        edge_attr_tensor = torch.tensor(Jij[i], dtype=torch.float).unsqueeze(1)
        
        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr_tensor,
        )
        data_list.append(data)
    
    return data_list


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load model from checkpoint.
    
    Supports:
        - Framework format: 'config' and 'model_state_dict' keys
        - Old format: 'model_state_dict' and 'model_kwargs' keys
        - Old GNN architecture (with LayerNorm in edge_decoder)
        - New GNN architecture (without LayerNorm)
    """
    # Add src to path for model imports
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get state dict (same key in all formats)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model")
    if state_dict is None:
        raise ValueError(f"No model weights found. Keys: {list(ckpt.keys())}")
    
    # Detect if this is a GNN
    is_gnn = any("node_encoder" in k or "edge_decoder" in k for k in state_dict.keys())
    
    if is_gnn:
        # Infer architecture from state_dict
        hidden_dim = state_dict.get("node_encoder.weight", torch.zeros(128, 1)).shape[0]
        num_layers = sum(1 for k in state_dict if k.startswith("layers.") and k.endswith(".msg_mlp.0.weight"))
        
            
        model = GNN(
            node_dim=1,
            edge_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="silu",
        )
        
        cfg = {"model": {"name": "gnn", "hidden_dim": hidden_dim, "num_layers": num_layers}}
        
        model.load_state_dict(state_dict)
    
    else: # this reads the MLP case and handles different pytorch loading cases
        # MLP
        if "config" in ckpt or "cfg" in ckpt:
            cfg = ckpt.get("config") or ckpt.get("cfg")
            model = MLP(
                input_dim=cfg["data"].get("input_dim", 10),
                output_dim=cfg["data"].get("output_dim", 6),
                hidden_dims=cfg["model"].get("hidden_dims", [128, 256, 128]),
                activation=cfg["model"].get("activation", "silu"),
            )
        elif "model_kwargs" in ckpt:
            model_kwargs = ckpt["model_kwargs"]
            model = MLP(
                input_dim=model_kwargs.get("input_dim", 10),
                output_dim=model_kwargs.get("output_dim", 6),
                hidden_dims=model_kwargs.get("hidden_dims", [128, 256, 128]),
                activation="silu",
            )
            cfg = {"model": {"name": "mlp", **model_kwargs}}
        else:
            model = MLP(
                input_dim=10,
                output_dim=6,
                hidden_dims=[128, 256, 128],
                activation="silu",
            )
            cfg = {"model": {"name": "mlp"}}
        
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, cfg


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, xi_i, Jij_i, n_qubits: int = 4, device: str = "cpu", is_gnn: bool = True):
    """
    Run model inference.
    
    Args:
        model: Loaded model (GNN or MLP)
        xi_i: (N, n_qubits) array of single-site expectation values
        Jij_i: (n_edges,) array of coupling values (same for all samples)
        n_qubits: Number of qubits
        device: Device to run on
        is_gnn: Whether model is GNN (True) or MLP (False)
    
    Returns:
        ZiZj_pred: (N, n_edges) predicted correlations
        O_pred: (N,) predicted observable
    """
    N = xi_i.shape[0]
    n_edges = math.comb(n_qubits, 2)
    
    # Convert xi_i to X_i
    X_i = 2 * xi_i
    
    # Tile Jij for all samples
    Jij = np.tile(Jij_i, (N, 1))
    
    if is_gnn:
        # GNN path - use graph data
        edges = list(itertools.combinations(range(n_qubits), 2))
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        data_list = create_graph_dataset(X_i, Jij, edge_index)
        loader = DataLoader(data_list, batch_size=256, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                preds.append(pred.cpu())
        
        preds = torch.cat(preds, dim=0)
        ZiZj_pred = preds.view(N, n_edges).numpy()
    
    else:
        # MLP path - use flat tensor input [X_i, Jij]
        X_flat = np.hstack([X_i, Jij])  # (N, n_qubits + n_edges)
        X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            preds = model(X_tensor)
        
        ZiZj_pred = preds.cpu().numpy()
    
    # Compute O_pred = 0.25 * sum(Jij * ZiZj)
    O_pred = 0.25 * np.sum(Jij * ZiZj_pred, axis=1)
    
    return ZiZj_pred, O_pred


# =============================================================================
# Plotting
# =============================================================================

def plot_points(x_data, y_data, dot_offset=0):
    """Plot data points on canvas."""
    for x, y in zip(x_data, y_data):
        # Map to effective coordinates
        ex = int((x - x_min) / (x_max - x_min) * (eff_width - 1))
        ey = int((y - y_min) / (y_max - y_min) * (eff_height - 1))
        ey = eff_height - 1 - ey  # Flip y
        
        # Map to character coordinates
        cx = ex // 2
        cy = ey // 4
        
        # Dot position within character
        dx = ex % 2
        dy = ey % 4
        
        if 0 <= cx < width and 0 <= cy < height:
            canvas[cy][cx] |= DOT_MAP[dy][dx]

def plot_terminal(xi_avg, O_truth, O_pred, r_value, title="O vs ξ", width=80, height=25):
    """
    Implementation of a Unicode terminal plot, based on:
    https://github.com/asciimoo/drawille
    
    Uses Braille characters for fine resolution (2x4 dots per character = 160x100 effective resolution).
    """
    # Braille base and dot positions
    # Braille char = 0x2800 + dot pattern
    # Dots: 1 4
    #       2 5
    #       3 6
    #       7 8
    BRAILLE_BASE = 0x2800
    DOT_MAP = [
        [0x01, 0x08],  # row 0
        [0x02, 0x10],  # row 1
        [0x04, 0x20],  # row 2
        [0x40, 0x80],  # row 3
    ]
    
    # Effective resolution
    eff_width = width * 2
    eff_height = height * 4
    
    # Data bounds
    x_min, x_max = xi_avg.min(), xi_avg.max()
    
    all_y = O_pred.copy()
    if O_truth is not None:
        all_y = np.concatenate([O_pred, O_truth])
    y_min, y_max = all_y.min(), all_y.max()
    
    # Add padding
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad
    
    # Initialize canvas
    canvas = [[0 for _ in range(width)] for _ in range(height)]
    
    # Plot prediction (all dots)
    plot_points(xi_avg, O_pred)
    
    # Create output with colors
    BLUE = '\033[94m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # If we have truth, create separate canvas for it
    if O_truth is not None:
        canvas_truth = [[0 for _ in range(width)] for _ in range(height)]
        canvas_pred = [[0 for _ in range(width)] for _ in range(height)]
        
        # Plot on separate canvases
        for x, y in zip(xi_avg, O_truth):
            ex = int((x - x_min) / (x_max - x_min) * (eff_width - 1))
            ey = int((y - y_min) / (y_max - y_min) * (eff_height - 1))
            ey = eff_height - 1 - ey
            cx, cy = ex // 2, ey // 4
            dx, dy = ex % 2, ey % 4
            if 0 <= cx < width and 0 <= cy < height:
                canvas_truth[cy][cx] |= DOT_MAP[dy][dx]
        
        for x, y in zip(xi_avg, O_pred):
            ex = int((x - x_min) / (x_max - x_min) * (eff_width - 1))
            ey = int((y - y_min) / (y_max - y_min) * (eff_height - 1))
            ey = eff_height - 1 - ey
            cx, cy = ex // 2, ey // 4
            dx, dy = ex % 2, ey % 4
            if 0 <= cx < width and 0 <= cy < height:
                canvas_pred[cy][cx] |= DOT_MAP[dy][dx]
    
    # Build output
    lines = []
    
    # Title
    title_str = f"{title} (r={r_value})"
    lines.append(f"{BOLD}{title_str:^{width+12}}{RESET}")
    lines.append("")
    
    # Y-axis label positions
    y_labels = [y_max, (y_max + y_min) / 2, y_min]
    y_label_rows = [0, height // 2, height - 1]
    
    # Plot area
    for row in range(height):
        # Y-axis label
        if row in y_label_rows:
            label = f"{y_labels[y_label_rows.index(row)]:>7.3f} │"
        else:
            label = "        │"
        
        # Plot content
        if O_truth is not None:
            row_chars = []
            for col in range(width):
                truth_dots = canvas_truth[row][col]
                pred_dots = canvas_pred[row][col]
                
                if truth_dots and pred_dots:
                    # Overlap - show combined in green
                    combined = truth_dots | pred_dots
                    row_chars.append(f"{GREEN}{chr(BRAILLE_BASE + combined)}{RESET}")
                elif truth_dots:
                    row_chars.append(f"{RED}{chr(BRAILLE_BASE + truth_dots)}{RESET}")
                elif pred_dots:
                    row_chars.append(f"{BLUE}{chr(BRAILLE_BASE + pred_dots)}{RESET}")
                else:
                    row_chars.append(" ")
            lines.append(label + "".join(row_chars))
        else:
            row_chars = []
            for col in range(width):
                dots = canvas[row][col]
                if dots:
                    row_chars.append(f"{BLUE}{chr(BRAILLE_BASE + dots)}{RESET}")
                else:
                    row_chars.append(" ")
            lines.append(label + "".join(row_chars))
    
    # X-axis
    lines.append("        └" + "─" * width)
    
    # X-axis labels
    x_label_line = f"        {x_min:<10.3f}" + " " * (width - 22) + f"{x_max:>10.3f}"
    lines.append(x_label_line)
    
    # X-axis title
    lines.append(f"{'<ξ>':^{width + 10}}")
    
    # Legend
    lines.append("")
    if O_truth is not None:
        lines.append(f"  {RED}●{RESET} truth    {BLUE}●{RESET} pred    {GREEN}●{RESET} overlap")
    else:
        lines.append(f"  {BLUE}●{RESET} pred")
    
    # Stats
    lines.append("")
    lines.append(f"  {GRAY}O_pred:  min={O_pred.min():.4f}  max={O_pred.max():.4f}  mean={O_pred.mean():.4f}{RESET}")
    if O_truth is not None:
        diff = O_pred - O_truth
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        lines.append(f"  {GRAY}O_truth: min={O_truth.min():.4f}  max={O_truth.max():.4f}  mean={O_truth.mean():.4f}{RESET}")
        lines.append(f"  {GRAY}Error:   MSE={mse:.6f}  MAE={mae:.6f}{RESET}")
    
    # Print
    print("\n".join(lines))


def save_csv(xi_avg, O_truth, O_pred, r_value, save_path: str):
    """
    Save results to CSV file for plotting elsewhere.
    
    Args:
        xi_avg: (N,) x-axis values
        O_truth: (N,) ground truth (or None if not available)
        O_pred: (N,) prediction
        r_value: r parameter value
        save_path: Path to save CSV
    """
    with open(save_path, 'w') as f:
        if O_truth is not None:
            f.write("xi_avg,O_truth,O_pred\n")
            for xi, ot, op in zip(xi_avg, O_truth, O_pred):
                f.write(f"{xi},{ot},{op}\n")
        else:
            f.write("xi_avg,O_pred\n")
            for xi, op in zip(xi_avg, O_pred):
                f.write(f"{xi},{op}\n")
    
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VDAT-ML Inference")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument("dat_dir", type=str, help="Directory containing output_*.dat files")
    parser.add_argument("--r", type=float, nargs="+", default=[0, 0.05, 0.25],
                        help="r values to plot (default: 0 0.05 0.25)")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output directory for PNGs (default: same as checkpoint)")
    parser.add_argument("--save-only", action="store_true", 
                        help="Skip terminal plot, only save PNG")
    parser.add_argument("--terminal-only", action="store_true",
                        help="Skip PNG, only show terminal plot (avoids matplotlib)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Paths
    ckpt_path = Path(args.checkpoint)
    
    # Output directory for plots
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data dir:   {args.dat_dir}")
    print(f"r values:   {args.r}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Load model
    print("Loading model...")
    model, cfg = load_model(str(ckpt_path), args.device)
    print(f"  Model: {cfg['model']['name']}")
    print()
    
    # Mapping of r values to dat files
    # Files: output_0.dat, output_0.05.dat, output_0.25.dat
    dat_dir = Path(args.dat_dir)
    dat_files = {
        0: dat_dir / "output_0.dat",
        0.05: dat_dir / "output_0.05.dat",
        0.25: dat_dir / "output_0.25.dat",
    }
    
    # Determine if model is GNN or MLP
    model_name = cfg.get("model", {}).get("name", "gnn")
    is_gnn = model_name == "gnn"
    print(f"Model type: {'GNN' if is_gnn else 'MLP'}")
    print()
    
    # Process each r value
    for r in args.r:
        print(f"Processing r={r}...")
        
        # Get dat file for this r
        dat_file = dat_files.get(r)
        
        if dat_file is None or not Path(dat_file).exists():
            print(f"  ERROR: dat file not found: {dat_file}")
            continue
        
        # Load data with truth
        theta, xi_avg, O_truth, xi_i = load_dat_file(str(dat_file), args.n_qubits)
        print(f"  Samples: {len(theta)}")
        print(f"  O_truth range: [{O_truth.min():.4f}, {O_truth.max():.4f}]")
        
        # Run inference
        Jij_single = compute_Jij(r, args.n_qubits)
        ZiZj_pred, O_pred = run_inference(model, xi_i, Jij_single, args.n_qubits, args.device, is_gnn)
        print(f"  O_pred range: [{O_pred.min():.4f}, {O_pred.max():.4f}]")
        
        # Always print error stats
        diff = O_pred - O_truth
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        print(f"  Error:   MSE={mse:.6f}  MAE={mae:.6f}")
        
        # Save CSV
        csv_path = output_dir / f"{ckpt_path.stem}_r{r}.csv"
        title = f"O vs <ξ> - {ckpt_path.stem}"
        
        if not args.terminal_only:
            save_csv(xi_avg, O_truth, O_pred, r, str(csv_path))
        
        if not args.save_only:
            plot_terminal(xi_avg, O_truth, O_pred, r, title)
        
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()
