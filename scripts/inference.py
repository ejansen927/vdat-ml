#!/usr/bin/env python3
"""
Inference script for VDAT-ML.

Loads a trained model checkpoint, reads a .dat file from VDAT,
and plots O_truth vs O_pred for multiple r values.

Supports both legacy and current GNN architectures via auto-detection.

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
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj


# =============================================================================
# Activation / Normalization Helpers
# =============================================================================

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    if name.lower() not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name.lower()]()


def get_norm_layer(name: str, dim: int) -> Optional[nn.Module]:
    """Get normalization layer by name."""
    if name is None or name.lower() == "none":
        return None
    elif name.lower() == "layer_norm":
        return nn.LayerNorm(dim)
    elif name.lower() == "batch_norm":
        return nn.BatchNorm1d(dim)
    else:
        raise ValueError(f"Unknown norm: {name}")


# =============================================================================
# Legacy GNN (for old checkpoints)
# =============================================================================

class LegacyInteractiveNet(nn.Module):
    """
    Legacy Interactive Network layer (no edge updates, no gating).
    Used for checkpoints from ~Jan 2026 and earlier.
    """
    
    def __init__(self, hidden_dim: int, activation: str = "silu"):
        super().__init__()
        
        act_fn = get_activation(activation)
        
        # Message MLP: [h_target, h_source, J_ij] -> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn,
        )
        
        # Update MLP: [h_i, aggregated_messages] -> delta h_i
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn,
        )
    
    def forward(self, h: torch.Tensor, adj_J: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Node embeddings, shape (B, N, D)
            adj_J: Adjacency matrix with J values, shape (B, N, N)
            
        Returns:
            Updated node embeddings, shape (B, N, D)
        """
        B, N, D = h.shape
        
        # Expand for pairwise operations
        h_source = h.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, D)
        h_target = h.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, D)
        J_expanded = adj_J.unsqueeze(-1)              # (B, N, N, 1)
        
        # Compute messages
        raw_msgs = torch.cat([h_target, h_source, J_expanded], dim=-1)
        messages = self.msg_mlp(raw_msgs)
        
        # Mask self-messages
        mask = 1 - torch.eye(N, device=h.device).view(1, N, N, 1)
        messages = messages * mask
        
        # Aggregate messages (sum over source nodes)
        agg_msgs = torch.sum(messages, dim=2)
        
        # Update nodes with residual
        update_input = torch.cat([h, agg_msgs], dim=-1)
        dh = self.update_mlp(update_input)
        
        return h + dh


class LegacyGNN(nn.Module):
    """
    Legacy Graph Neural Network (no edge encoder, node-pair edge decoder).
    Used for checkpoints from ~Jan 2026 and earlier.
    
    Architecture:
        1. Node encoder: Linear(node_dim -> hidden_dim)
        2. N LegacyInteractiveNet layers for message passing
        3. Edge decoder: MLP([h_src, h_dst, J_ij] -> prediction)
    """
    
    def __init__(
        self,
        node_dim: int = 1,
        edge_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "silu",
        **kwargs,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        act_fn = get_activation(activation)
        
        # Node encoder
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList([
            LegacyInteractiveNet(hidden_dim, activation)
            for _ in range(num_layers)
        ])
        
        # Edge decoder (node-pair based)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Bounded output [-1, 1]
        )
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass."""
        # Convert to dense batch format
        x_dense, mask = to_dense_batch(data.x, data.batch)  # (B, N, node_dim)
        
        # Encode nodes
        h = self.node_encoder(x_dense)  # (B, N, hidden_dim)
        
        # Build dense adjacency with J values
        adj_J = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze(-1)
        # Make symmetric (J_ij = J_ji)
        adj_J = adj_J + adj_J.transpose(1, 2)
        
        # Message passing
        for layer in self.layers:
            h = layer(h, adj_J)
        
        # Decode edges
        batch_idx = data.batch[data.edge_index[0]]
        
        # Convert global indices to local (per-graph) indices
        N = x_dense.size(1)
        src_local = data.edge_index[0] % N
        dst_local = data.edge_index[1] % N
        
        # Gather node embeddings for each edge
        h_src = h[batch_idx, src_local]
        h_dst = h[batch_idx, dst_local]
        
        # Concatenate with edge features and decode
        edge_input = torch.cat([h_src, h_dst, data.edge_attr], dim=-1)
        out = self.edge_decoder(edge_input)
        return out.squeeze(-1)


# =============================================================================
# Current GNN (for new checkpoints)
# =============================================================================

class CurrentInteractiveNet(nn.Module):
    """
    Current Interactive Network layer with edge updates and gating.
    Used for checkpoints from ~Feb 2026 onwards.
    """
    
    def __init__(self, hidden_dim: int, activation: str = "silu", pooling: str = "mean", **kwargs):
        super().__init__()
        
        act_fn = get_activation(activation)
        self.pooling = pooling
        
        # Message MLP: [h_target, h_source, e] -> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            act_fn,
        )
        
        # Update MLP: [h_i, aggregated_messages] -> delta h_i
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        # Edge update
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor):
        """
        Forward pass.
        
        Args:
            h: Node embeddings, shape (B, N, D)
            e: Edge embeddings, shape (B, N, N, D)
            
        Returns:
            Updated node embeddings (B, N, D) and edge embeddings (B, N, N, D)
        """
        B, N, D = h.shape
        
        # Expand for pairwise operations
        h_source = h.unsqueeze(1).repeat(1, N, 1, 1)
        h_target = h.unsqueeze(2).repeat(1, 1, N, 1)

        # Edge update
        h_sum_e = h_source + h_target
        h_diff_e = torch.abs(h_source - h_target)
        h_prod_e = h_source * h_target
        edge_input = torch.cat([h_sum_e, h_diff_e, h_prod_e, e], dim=-1)
        de = self.edge_update_mlp(edge_input)
        edge_gate = torch.sigmoid(self.edge_gate(torch.cat([e, de], dim=-1)))
        e = edge_gate * e + (1 - edge_gate) * de
        
        # Messages
        combined_msg = torch.cat([h_source, h_target, e], dim=-1)
        messages = self.msg_mlp(combined_msg)
        
        # Mask self messages
        mask = 1 - torch.eye(N, device=h.device).view(1, N, N, 1)
        messages = messages * mask
        
        # Pool messages
        if self.pooling == "sum":
            agg_msgs = torch.sum(messages, dim=2)
        elif self.pooling == "mean":
            agg_msgs = torch.sum(messages, dim=2) / (N - 1)
        elif self.pooling == "max":
            agg_msgs = torch.max(messages, dim=2).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Update nodes with gating
        update_input = torch.cat([h, agg_msgs], dim=-1)
        dh = self.update_mlp(update_input)
        gate = torch.sigmoid(self.gate_layer(torch.cat([h, dh], dim=-1)))
        h = gate * h + (1 - gate) * dh

        return h, e


class CurrentGNN(nn.Module):
    """
    Current Graph Neural Network with edge encoder and edge-based decoder.
    Used for checkpoints from ~Feb 2026 onwards.
    """
    
    def __init__(
        self,
        node_dim: int = 1,
        edge_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "silu",
        pooling: str = "mean",
        **kwargs,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        act_fn = get_activation(activation)
        
        # Node encoder
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Message passing layers
        self.layers = nn.ModuleList([
            CurrentInteractiveNet(hidden_dim, activation, pooling)
            for _ in range(num_layers)
        ])
        
        # Edge decoder (edge-state based)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass."""
        # Convert to dense batch format
        x_dense, mask = to_dense_batch(data.x, data.batch)
        
        # Encode nodes
        h = self.node_encoder(x_dense)
        
        # Build dense adjacency with J values
        adj_J = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze(-1)
        adj_J = adj_J + adj_J.transpose(1, 2)
        
        # Encode edges
        e = self.edge_encoder(adj_J.unsqueeze(-1))

        # Message passing
        for layer in self.layers:
            h, e = layer(h, e)
        
        # Decode edges
        batch_idx = data.batch[data.edge_index[0]]
        N = x_dense.size(1)
        src_local = data.edge_index[0] % N
        dst_local = data.edge_index[1] % N

        # Get edge embeddings
        e_ij = e[batch_idx, src_local, dst_local]
        edge_input = torch.cat([e_ij, data.edge_attr], dim=-1)

        out = self.edge_decoder(edge_input)
        return out.squeeze(-1)


# =============================================================================
# MLP (unchanged)
# =============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron for tensor data."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 256, 128],
        activation: str = "silu",
        norm: str = "layer_norm",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            norm_layer = get_norm_layer(norm, hidden_dim)
            if norm_layer is not None:
                layers.append(norm_layer)
            
            layers.append(get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Data Loading
# =============================================================================

def load_dat_file(filepath: str, n_qubits: int = 4):
    """
    Load VDAT output .dat file.
    
    Format: θ, ξ_avg, O_truth, ξ_i[0], ξ_i[1], ..., ξ_i[n_qubits-1]
    """
    data = np.loadtxt(filepath)
    
    theta = data[:, 0]
    xi_avg = data[:, 1]
    O_truth = data[:, 2]
    xi_i = data[:, 3:3 + n_qubits]
    
    return theta, xi_avg, O_truth, xi_i


def compute_Jij(r: float, n_qubits: int = 4):
    """Compute Jij coupling values for given r parameter."""
    if n_qubits != 4:
        raise NotImplementedError("Only 4 qubits supported currently")
    
    J12, J34 = 1.0, 1.0
    J14, J23 = (1 - 2*r), (1 - 2*r)
    J13, J24 = (1 - 3*r), (1 - 3*r)
    
    return np.array([J12, J13, J14, J23, J24, J34])


def create_graph_dataset(Xi, Jij, edge_index, dtype=torch.float64):
    """Create PyG Data objects from Xi and Jij arrays."""
    data_list = []
    num_samples = Xi.shape[0]
    
    for i in range(num_samples):
        x_tensor = torch.tensor(Xi[i], dtype=dtype).unsqueeze(1)
        edge_attr_tensor = torch.tensor(Jij[i], dtype=dtype).unsqueeze(1)
        
        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr_tensor,
        )
        data_list.append(data)
    
    return data_list


# =============================================================================
# Model Loading (with architecture auto-detection)
# =============================================================================

def detect_gnn_version(state_dict: dict) -> str:
    """
    Detect GNN architecture version from state_dict keys.
    
    Returns:
        'legacy' - Old architecture (Jan 2026 and earlier)
        'current' - New architecture (Feb 2026 onwards)
    """
    # Current version has edge_encoder
    has_edge_encoder = any("edge_encoder" in k for k in state_dict.keys())
    
    # Current version has edge_update_mlp in layers
    has_edge_update = any("edge_update_mlp" in k for k in state_dict.keys())
    
    # Current version has gate_layer in layers
    has_gate = any("gate_layer" in k for k in state_dict.keys())
    
    # Check msg_mlp input dimension (legacy: 2*D+1, current: 3*D)
    msg_weight_key = "layers.0.msg_mlp.0.weight"
    if msg_weight_key in state_dict:
        msg_input_dim = state_dict[msg_weight_key].shape[1]
        hidden_dim = state_dict["node_encoder.weight"].shape[0]
        
        # Legacy: 2*hidden + 1, Current: 3*hidden
        expected_legacy = hidden_dim * 2 + 1
        expected_current = hidden_dim * 3
        
        if msg_input_dim == expected_legacy:
            return "legacy"
        elif msg_input_dim == expected_current:
            return "current"
    
    # Fallback to feature detection
    if has_edge_encoder or has_edge_update or has_gate:
        return "current"
    else:
        return "legacy"


def infer_gnn_params(state_dict: dict) -> dict:
    """Infer GNN parameters from state_dict shapes."""
    hidden_dim = state_dict["node_encoder.weight"].shape[0]
    node_dim = state_dict["node_encoder.weight"].shape[1]
    
    num_layers = sum(
        1 for k in state_dict 
        if k.startswith("layers.") and k.endswith(".msg_mlp.0.weight")
    )
    
    return {
        "hidden_dim": hidden_dim,
        "node_dim": node_dim,
        "edge_dim": 1,  # Always 1 for our use case
        "num_layers": num_layers,
    }


def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load model from checkpoint with auto-detection of architecture version.
    
    Supports:
        - Legacy GNN (Jan 2026 and earlier)
        - Current GNN (Feb 2026 onwards)
        - MLP
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get state dict
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model")
    if state_dict is None:
        raise ValueError(f"No model weights found. Keys: {list(ckpt.keys())}")
    
    # Get config if available
    cfg = ckpt.get("config") or ckpt.get("cfg") or {}
    
    # Detect model type
    is_gnn = any("node_encoder" in k or "edge_decoder" in k for k in state_dict.keys())
    
    if is_gnn:
        # Detect version
        version = detect_gnn_version(state_dict)
        params = infer_gnn_params(state_dict)
        
        print(f"  Detected GNN version: {version}")
        print(f"  Parameters: hidden_dim={params['hidden_dim']}, num_layers={params['num_layers']}")
        
        # Get additional params from config if available
        model_cfg = cfg.get("model", {})
        activation = model_cfg.get("activation", "silu")
        pooling = model_cfg.get("pooling", "mean")
        
        if version == "legacy":
            model = LegacyGNN(
                node_dim=params["node_dim"],
                edge_dim=params["edge_dim"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                activation=activation,
            )
        else:  # current
            model = CurrentGNN(
                node_dim=params["node_dim"],
                edge_dim=params["edge_dim"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                activation=activation,
                pooling=pooling,
            )
        
        # Ensure config reflects what we loaded
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["name"] = "gnn"
        cfg["model"]["version"] = version
        cfg["model"]["hidden_dim"] = params["hidden_dim"]
        cfg["model"]["num_layers"] = params["num_layers"]
        
    else:
        # MLP
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        
        if "hidden_dims" in model_cfg:
            hidden_dims = list(model_cfg["hidden_dims"])
        elif "model_kwargs" in ckpt:
            hidden_dims = ckpt["model_kwargs"].get("hidden_dims", [128, 256, 128])
        else:
            # Infer from state_dict
            hidden_dims = []
            i = 0
            while f"net.{i}.weight" in state_dict:
                hidden_dims.append(state_dict[f"net.{i}.weight"].shape[0])
                i += 3  # Linear + Norm + Activation
            hidden_dims = hidden_dims[:-1] if hidden_dims else [128, 256, 128]
        
        input_dim = data_cfg.get("input_dim", 10)
        output_dim = data_cfg.get("output_dim", 6)
        
        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=model_cfg.get("activation", "silu"),
            norm=model_cfg.get("norm", "layer_norm"),
            dropout=model_cfg.get("dropout", 0.0),
        )
        
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["name"] = "mlp"
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning - Missing keys: {missing}")
    if unexpected:
        print(f"  Warning - Unexpected keys: {unexpected}")
    
    # Detect dtype from weights
    sample_weight = next(iter(state_dict.values()))
    model_dtype = sample_weight.dtype
    print(f"  Model dtype: {model_dtype}")
    
    model.to(device)
    model.eval()
    
    return model, cfg


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, xi_i, Jij_i, n_qubits: int = 4, device: str = "cpu", is_gnn: bool = True):
    """Run model inference."""
    N = xi_i.shape[0]
    n_edges = math.comb(n_qubits, 2)
    
    # Convert xi_i to X_i
    X_i = 2 * xi_i
    
    # Tile Jij for all samples
    Jij = np.tile(Jij_i, (N, 1))
    
    # Detect model dtype
    model_dtype = next(model.parameters()).dtype
    
    if is_gnn:
        # GNN path
        edges = list(itertools.combinations(range(n_qubits), 2))
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        data_list = create_graph_dataset(X_i, Jij, edge_index, dtype=model_dtype)
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
        # MLP path
        X_flat = np.hstack([X_i, Jij])
        X_tensor = torch.tensor(X_flat, dtype=model_dtype).to(device)
        
        with torch.no_grad():
            preds = model(X_tensor)
        
        ZiZj_pred = preds.cpu().numpy()
    
    # Compute O_pred = 0.25 * sum(Jij * ZiZj)
    O_pred = 0.25 * np.sum(Jij * ZiZj_pred, axis=1)
    
    return ZiZj_pred, O_pred


# =============================================================================
# Plotting
# =============================================================================

def plot_terminal(xi_avg, O_truth, O_pred, r_value, title="O vs ξ", width=80, height=25):
    """
    Unicode terminal plot using Braille characters.
    """
    BRAILLE_BASE = 0x2800
    DOT_MAP = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]
    
    eff_width = width * 2
    eff_height = height * 4
    
    x_min, x_max = xi_avg.min(), xi_avg.max()
    
    all_y = O_pred.copy()
    if O_truth is not None:
        all_y = np.concatenate([O_pred, O_truth])
    y_min, y_max = all_y.min(), all_y.max()
    
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad
    
    BLUE = '\033[94m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    canvas_truth = [[0 for _ in range(width)] for _ in range(height)]
    canvas_pred = [[0 for _ in range(width)] for _ in range(height)]
    
    def plot_to_canvas(canvas, x_data, y_data):
        for x, y in zip(x_data, y_data):
            if x_max == x_min:
                ex = eff_width // 2
            else:
                ex = int((x - x_min) / (x_max - x_min) * (eff_width - 1))
            if y_max == y_min:
                ey = eff_height // 2
            else:
                ey = int((y - y_min) / (y_max - y_min) * (eff_height - 1))
            ey = eff_height - 1 - ey
            cx, cy = ex // 2, ey // 4
            dx, dy = ex % 2, ey % 4
            if 0 <= cx < width and 0 <= cy < height:
                canvas[cy][cx] |= DOT_MAP[dy][dx]
    
    if O_truth is not None:
        plot_to_canvas(canvas_truth, xi_avg, O_truth)
    plot_to_canvas(canvas_pred, xi_avg, O_pred)
    
    lines = []
    title_str = f"{title} (r={r_value})"
    lines.append(f"{BOLD}{title_str:^{width+12}}{RESET}")
    lines.append("")
    
    y_labels = [y_max, (y_max + y_min) / 2, y_min]
    y_label_rows = [0, height // 2, height - 1]
    
    for row in range(height):
        if row in y_label_rows:
            label = f"{y_labels[y_label_rows.index(row)]:>7.3f} │"
        else:
            label = "        │"
        
        row_chars = []
        for col in range(width):
            truth_dots = canvas_truth[row][col] if O_truth is not None else 0
            pred_dots = canvas_pred[row][col]
            
            if truth_dots and pred_dots:
                combined = truth_dots | pred_dots
                row_chars.append(f"{GREEN}{chr(BRAILLE_BASE + combined)}{RESET}")
            elif truth_dots:
                row_chars.append(f"{RED}{chr(BRAILLE_BASE + truth_dots)}{RESET}")
            elif pred_dots:
                row_chars.append(f"{BLUE}{chr(BRAILLE_BASE + pred_dots)}{RESET}")
            else:
                row_chars.append(" ")
        lines.append(label + "".join(row_chars))
    
    lines.append("        └" + "─" * width)
    x_label_line = f"        {x_min:<10.3f}" + " " * (width - 22) + f"{x_max:>10.3f}"
    lines.append(x_label_line)
    lines.append(f"{'<ξ>':^{width + 10}}")
    
    lines.append("")
    if O_truth is not None:
        lines.append(f"  {RED}●{RESET} truth    {BLUE}●{RESET} pred    {GREEN}●{RESET} overlap")
    else:
        lines.append(f"  {BLUE}●{RESET} pred")
    
    lines.append("")
    lines.append(f"  {GRAY}O_pred:  min={O_pred.min():.4f}  max={O_pred.max():.4f}  mean={O_pred.mean():.4f}{RESET}")
    if O_truth is not None:
        diff = O_pred - O_truth
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        lines.append(f"  {GRAY}O_truth: min={O_truth.min():.4f}  max={O_truth.max():.4f}  mean={O_truth.mean():.4f}{RESET}")
        lines.append(f"  {GRAY}Error:   MSE={mse:.6f}  MAE={mae:.6f}{RESET}")
    
    print("\n".join(lines))


def save_csv(xi_avg, O_truth, O_pred, r_value, save_path: str):
    """Save results to CSV file."""
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
                        help="Output directory for CSVs (default: same as checkpoint)")
    parser.add_argument("--save-only", action="store_true", 
                        help="Skip terminal plot, only save CSV")
    parser.add_argument("--terminal-only", action="store_true",
                        help="Skip CSV, only show terminal plot")
    return parser.parse_args()


def main():
    args = parse_args()
    
    ckpt_path = Path(args.checkpoint)
    
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
    
    print("Loading model...")
    model, cfg = load_model(str(ckpt_path), args.device)
    print(f"  Model: {cfg['model']['name']}")
    print()
    
    dat_dir = Path(args.dat_dir)
    dat_files = {
        0: dat_dir / "output_0.dat",
        0.05: dat_dir / "output_0.05.dat",
        0.25: dat_dir / "output_0.25.dat",
    }
    
    model_name = cfg.get("model", {}).get("name", "gnn")
    is_gnn = model_name == "gnn"
    print(f"Model type: {'GNN' if is_gnn else 'MLP'}")
    if is_gnn:
        print(f"GNN version: {cfg['model'].get('version', 'unknown')}")
    print()
    
    for r in args.r:
        print(f"Processing r={r}...")
        
        dat_file = dat_files.get(r)
        
        if dat_file is None or not Path(dat_file).exists():
            print(f"  ERROR: dat file not found: {dat_file}")
            continue
        
        theta, xi_avg, O_truth, xi_i = load_dat_file(str(dat_file), args.n_qubits)
        print(f"  Samples: {len(theta)}")
        print(f"  O_truth range: [{O_truth.min():.4f}, {O_truth.max():.4f}]")
        
        Jij_single = compute_Jij(r, args.n_qubits)
        ZiZj_pred, O_pred = run_inference(model, xi_i, Jij_single, args.n_qubits, args.device, is_gnn)
        print(f"  O_pred range: [{O_pred.min():.4f}, {O_pred.max():.4f}]")
        
        diff = O_pred - O_truth
        mse = np.mean(diff ** 2)
        mae = np.mean(np.abs(diff))
        print(f"  Error:   MSE={mse:.6f}  MAE={mae:.6f}")
        
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
