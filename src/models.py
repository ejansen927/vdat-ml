"""
Model definitions: MLP and GNN.

Both models:
    - Take dimensions from data (auto-detected)
    - Have configurable activation, normalization, dropout
    - End with Tanh() for bounded output [-1, 1]

MLP:
    - Takes raw tensors (batch, input_dim) -> (batch, output_dim)
    
GNN:
    - Takes PyG Data objects
    - Uses InteractiveNet message passing (custom architecture)
    - Outputs edge-level predictions
"""

import torch
import torch.nn as nn
from typing import List, Optional
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
        raise ValueError(f"Unknown norm: {name}. Choose from ['layer_norm', 'batch_norm', 'none']")


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tensor data.
    
    Takes (batch, input_dim) -> (batch, output_dim)
    
    Architecture defined by hidden_dims list:
        [128, 256, 128] -> 3 hidden layers with those widths
    
    Args:
        input_dim: Input feature dimension (auto-detected from data)
        output_dim: Output dimension (auto-detected from data)
        hidden_dims: List of hidden layer widths
        activation: Activation function name
        norm: Normalization type ('layer_norm', 'batch_norm', 'none')
        dropout: Dropout probability
    """
    
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Bounded output [-1, 1]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, input_dim)
            
        Returns:
            Output tensor, shape (batch, output_dim)
        """
        return self.net(x)


# =============================================================================
# GNN Components
# =============================================================================

class InteractiveNet(nn.Module):
    """
    Interactive Network layer for message passing.
    Based on DeepMind's Interaction Networks.
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


class GNN(nn.Module):
    """
    Graph Neural Network for quantum correlation prediction.
    
    Architecture:
        1. Node encoder: Linear(node_dim -> hidden_dim)
        2. N InteractiveNet layers for message passing (with dropout)
        3. Edge decoder: MLP([h_src, h_dst, J_ij] -> prediction)
    
    Output is edge-level predictions with Tanh activation (bounded [-1, 1]).
    """
    
    def __init__(
        self,
        node_dim: int = 1,
        edge_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "silu",
        norm: str = "layer_norm",  # kept for config compatibility
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        act_fn = get_activation(activation)
        
        # Node encoder
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Dropout layer (applied after encoder and each message passing layer)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Message passing layers
        self.layers = nn.ModuleList([
            InteractiveNet(hidden_dim, activation)
            for _ in range(num_layers)
        ])
        
        # Edge decoder with dropout
        if dropout > 0:
            self.edge_decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Tanh(),  # Bounded output [-1, 1]
            )
        else:
            self.edge_decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim // 2),
                act_fn,
                nn.Linear(hidden_dim // 2, 1),
                nn.Tanh(),  # Bounded output [-1, 1]
            )
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data/Batch with x, edge_index, edge_attr, batch
            
        Returns:
            Edge predictions, shape (num_edges_in_batch, 1)
        """
        # Convert to dense batch format
        x_dense, mask = to_dense_batch(data.x, data.batch)  # (B, N, node_dim)
        
        # Encode nodes
        h = self.node_encoder(x_dense)  # (B, N, hidden_dim)
        h = self.dropout_layer(h)
        
        # Build dense adjacency with J values
        adj_J = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze(-1)
        # Make symmetric (J_ij = J_ji)
        adj_J = adj_J + adj_J.transpose(1, 2)
        
        # Message passing with dropout after each layer
        for layer in self.layers:
            h = layer(h, adj_J)
            h = self.dropout_layer(h)
        
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
        return out.squeeze(-1)  # (num_edges,) to match target shape


# =============================================================================
# Model Builder
# =============================================================================

def build_model(cfg, dims: dict) -> nn.Module:
    """
    Build model from config and data dimensions.
    
    Args:
        cfg: Hydra config with cfg.model
        dims: Dict with input_dim, output_dim, etc. from setup_data()
        
    Returns:
        Instantiated model
    """
    model_name = cfg.model.name.lower()
    
    if model_name == "mlp":
        return MLP(
            input_dim=dims["input_dim"],
            output_dim=dims["output_dim"],
            hidden_dims=list(cfg.model.hidden_dims),
            activation=cfg.model.activation,
            norm=cfg.model.get("norm", "layer_norm"),
            dropout=cfg.model.get("dropout", 0.0),
        )
    
    elif model_name == "gnn":
        return GNN(
            node_dim=dims.get("node_dim", 1),
            edge_dim=dims.get("edge_dim", 1),
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            activation=cfg.model.activation,
            norm=cfg.model.get("norm", "layer_norm"),
            dropout=cfg.model.get("dropout", 0.0),
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from ['mlp', 'gnn']")
