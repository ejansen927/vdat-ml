"""
Model definitions: MLP and GNN.

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

# set to float64 (double), this is important
torch.set_default_dtype(torch.float64)

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
    """Add activation function by config file."""
    if name.lower() not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name.lower()]()


def get_norm_layer(name: str, dim: int) -> Optional[nn.Module]:
    """Add a normalization layer from config file."""
    if name is None or name.lower() == "none":
        return None
    elif name.lower() == "layer_norm":
        return nn.LayerNorm(dim)
    elif name.lower() == "batch_norm":
        return nn.BatchNorm1d(dim)
    else:
        raise ValueError(f"Current norm layers: {name}. Choose from ['layer_norm', 'batch_norm', 'none']")


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
    
    def __init__(self, hidden_dim: int, activation: str = "silu", pooling: str = "mean", **kwargs):
        super().__init__()
        
        act_fn = get_activation(activation) #SiLU seems best, set to default
        self.pooling = pooling
        #print(self.pooling)
        
        # Message MLP: [h_target, h_source, J_ij] -> message
        self.msg_mlp = nn.Sequential(
            #nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.Linear(hidden_dim * 3, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            act_fn,
        )
        
        # Update MLP: [h_i, aggregated_messages] -> delta h_i
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            act_fn,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        # lets try updating edges per layer too:
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    
    #def forward(self, h: torch.Tensor, adj_J: torch.Tensor) -> torch.Tensor:
    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor: # now takes edge embedding e
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
        h_target = h.unsqueeze(2).repeat(1, 1, N, 1)

        # more efficient? look into:
        #h_source = h.unsqueeze(1).expand(B, N, N, D)
        #h_target = h.unsqueeze(2).expand(B, N, N, D)

        #J_expanded = adj_J.unsqueeze(-1)              # (B, N, N, 1)
        
        # Compute messages
        #raw_msgs = torch.cat([h_target, h_source, J_expanded], dim=-1)
        ## gonna add an edge update before message passing:

        h_sum_e = h_source + h_target
        h_diff_e = torch.abs(h_source - h_target)
        h_prod_e = h_source * h_target
        edge_input = torch.cat([h_sum_e, h_diff_e, h_prod_e, e], dim=-1)
        de = self.edge_update_mlp(edge_input)
        edge_gate = torch.sigmoid(self.edge_gate(torch.cat([e,de],dim=-1)))
        e = edge_gate * e + (1 - edge_gate) * de
        # then continue as normal. trying to give edges as much info as possible, ie, sum and diff etc.
        #messages = e # pass e as the messages below, includes node info now
        
        combined_msg = torch.cat([h_source,h_target,e],dim=-1)
        messages = self.msg_mlp(combined_msg)

        # temporarily eliminate this node heavy message passing
        #raw_msgs = torch.cat([h_target, h_source, e], dim=-1)
        #messages = self.msg_mlp(raw_msgs)
        
        # remove self messages
        mask = 1 - torch.eye(N, device=h.device).view(1, N, N, 1)
        messages = messages * mask
        
        # pool messages:
        if self.pooling == "sum":
            agg_msgs = torch.sum(messages, dim=2)
        elif self.pooling == "mean":
            agg_msgs = torch.sum(messages, dim=2) / (N-1) # since Im masking i reduce N by 1
            #agg_msgs = torch.mean(messages,dim=2)
        elif self.pooling == "max":
            agg_msgs = torch.max(messages, dim=2).values
        else:
            raise ValueError(f"Unknown pooling method here, {self.pooling}")
        
        # Update nodes with residual
        update_input = torch.cat([h, agg_msgs], dim=-1)
        dh = self.update_mlp(update_input)
        
        # adding a gate here, testing:
        gate = torch.sigmoid(self.gate_layer(torch.cat([h, dh], dim=-1)))
        h = gate * h + (1 - gate) * dh

        #return h + dh
        #return self.norm(h + dh)
        #return h
        return h, e


class GNN(nn.Module):
    """
    Graph Neural Network for quantum correlation prediction.
    
    Architecture:
        1. Node encoder: Linear(node_dim -> hidden_dim)
        2. N InteractiveNet layers for message passing
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
        norm: str = "layer_norm",  # kept for config compatibility, not used
        dropout: float = 0.0,       # kept for config compatibility, not used
        pooling: str = "mean",
        **kwargs,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        act_fn = get_activation(activation)
        
        # Node encoder (pass the input nodes to a linear layer)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList([
            InteractiveNet(hidden_dim, activation,pooling)
            for _ in range(num_layers)
        ])

        # edge encoder here too
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Edge decoder
        self.edge_decoder = nn.Sequential(
            #nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.Linear(hidden_dim + 1, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, 1),
            ##nn.Tanh(),  # Bounded output [-1, 1]
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
        
        # Build dense adjacency with J values
        adj_J = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze(-1)

        # Make symmetric (J_ij = J_ji)
        adj_J = adj_J + adj_J.transpose(1, 2)
        
        e = self.edge_encoder(adj_J.unsqueeze(-1))

        # Message passing
        for layer in self.layers:
            #h = layer(h, adj_J)
            #h = layer(h, e)
            h, e = layer(h, e)
        
        # Decode edges
        batch_idx = data.batch[data.edge_index[0]]
        
        # Convert global indices to local (per-graph) indices
        N = x_dense.size(1)
        src_local = data.edge_index[0] % N
        dst_local = data.edge_index[1] % N

        # tmp remove this:
        # Gather node embeddings for each edge
        #h_src = h[batch_idx, src_local]
        #h_dst = h[batch_idx, dst_local]

        # add in some oeprations on h_src and dst
        #h_sum = h_src + h_dst
        #h_diff = torch.abs(h_src - h_dst)
        #h_diff = h_src * h_dst
        
        # Concatenate with edge features and decode
        #edge_input = torch.cat([h_src, h_dst, data.edge_attr], dim=-1)
        #edge_input = torch.cat([h_sum, h_diff, data.edge_attr], dim=-1)

        # try with edges:
        e_ij = e[batch_idx, src_local, dst_local]
        edge_input = torch.cat([e_ij, data.edge_attr],dim=-1)

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
            pooling=cfg.model.get("pooling", "mean"),
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from ['mlp', 'gnn']")
