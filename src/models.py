"""
Model definitions: MLP and GNN.

MLP:
    - simple architecture (batch, input_dim) -> (batch, output_dim)

GNN:
    - transforms data via PyG graph data objects, message passing -> output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch_geometric.utils import to_dense_batch, to_dense_adj

# set to float64 (double), this is important
torch.set_default_dtype(torch.float64)

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

        # possibly more efficient? look into later:
        #h_source = h.unsqueeze(1).expand(B, N, N, D)
        #h_target = h.unsqueeze(2).expand(B, N, N, D)

        #J_expanded = adj_J.unsqueeze(-1)              # (B, N, N, 1)

        # Compute messages
        #raw_msgs = torch.cat([h_target, h_source, J_expanded], dim=-1)
        ## gonna add an edge update before message passing:

        # add more pooled features like the sum and difference and product of e as a test
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

        # update nodes at each step
        update_input = torch.cat([h, agg_msgs], dim=-1)
        dh = self.update_mlp(update_input)

        # adding a gate here, testing: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
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
        norm: str = "layer_norm",
        dropout: float = 0.0,
        pooling: str = "mean",
        n_qubits_pe: int = 0, # add this for one hot encoding
        **kwargs,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_qubits_pe = n_qubits_pe

        act_fn = get_activation(activation)

        # one-hot: widen node encoder input when PE enabled
        encoder_input_dim = node_dim + (n_qubits_pe if n_qubits_pe > 0 else 0)  # one-hot
        self.node_encoder = nn.Linear(encoder_input_dim, hidden_dim)

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

    def _inject_node_pe(self, x_dense):
        """Just adding positions to nodes. Append one-hot positional encoding to node features.

        Node 0: [X0, 1, 0, 0, 0]
        Node 1: [X1, 0, 1, 0, 0]  etc.

        Breaks permutation equivariance so the GNN can
        distinguish nodes even when Xi and Jij are identical.
        """
        B, N, _ = x_dense.shape
        one_hot = torch.eye(
            N, dtype=x_dense.dtype, device=x_dense.device
        ).unsqueeze(0).expand(B, -1, -1)
        return torch.cat([x_dense, one_hot], dim=-1)

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data/Batch with x, edge_index, edge_attr, batch

        Returns:
            Edge predictions, shape (num_edges_in_batch, 1)
        """
        # convert to dense batch format
        x_dense, mask = to_dense_batch(data.x, data.batch)  # (B, N, node_dim)

        # one-hot: inject positional encoding before node encoder  # one-hot
        if self.n_qubits_pe > 0:                                   # one-hot
            x_dense = self._inject_node_pe(x_dense)                # one-hot: (B, N, node_dim+N)

        # encode the nodes to some h_dim dimensional vector
        h = self.node_encoder(x_dense)  # (B, N, hidden_dim)

        # build graph dense adjacency matrix using J values
        adj_J = to_dense_adj(data.edge_index, data.batch, data.edge_attr).squeeze(-1)

        # Make symmetric (J_ij = J_ji)
        adj_J = adj_J + adj_J.transpose(1, 2)

        e = self.edge_encoder(adj_J.unsqueeze(-1))

        # run L layers of message passing
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
        edge_input = torch.cat([e_ij, data.edge_attr],dim=-1) # concat updated edges with original edges, seems to make more stable

        out = self.edge_decoder(edge_input)

        # just add tanh back, stupid to remove it
        out = torch.tanh(out)

        return out.squeeze(-1)  # (num_edges,) to match target shape


# =============================================================================
# Dual-Head GNN
# =============================================================================

class DualHeadGNN(GNN):
    """                                                            #
    Dual-head GNN: shared backbone with separate magnitude and
    sign prediction heads.

    Motivation: sign errors seem to cause cause 60-80% of MSE in Ising cases so
    maybe we train them separately?

    Architecture:
        same exact as GNN, but at the end we split off:
        magnitude_head: e_ij, J_ij -> sigmoid -> |ZZ| in [0,1]   # predict abs(ZZ) now instead, same ending
        sign_head: e_ij, J_ij -> raw logit -> P(ZZ > 0)          # takes same output and passes to sigmoid [binary, ouputs 0 or 1]
        Inference: ZZ = (2*sigmoid(logit) - 1) * magnitude

    Returns dict with keys: magnitude, sign_logit, zz
    """

    def __init__(
        self,
        node_dim: int = 1,
        edge_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "silu",
        norm: str = "layer_norm",
        dropout: float = 0.0,
        pooling: str = "mean",
        n_qubits_pe: int = 0,                                     # one-hot: pass through to GNN
        **kwargs,
    ):
        # start by initializing og GNN (node_encoder, edge_encoder,
        # InteractiveNet layers, and edge_decoder which we won't
        # use but keep for checkpoint compat)
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            norm=norm,
            dropout=dropout,
            pooling=pooling,
            n_qubits_pe=n_qubits_pe,
            **kwargs,
        )

        act_fn = get_activation(activation)

        # (1) the magnitude head: predicts |ZZ| in [0, 1]
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # (2) this is the sign head: predicts raw logit for P(ZZ > 0)
        self.sign_head = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, 1),
        )

        # learnable loss weights , can be given as a parameter
        # these are optimized alongside model params
        self.log_sigma_mag = nn.Parameter(torch.zeros(1))
        self.log_sigma_sign = nn.Parameter(torch.zeros(1))

    def forward(self, data) -> dict:
        """
        Forward pass with dual-head output.

        Returns:
            dict with:
              magnitude: (num_edges,) in [0, 1]
              sign_logit: (num_edges,) raw logit
              zz: (num_edges,) reconstructed ZZ prediction
        """
        x_dense, mask = to_dense_batch(data.x, data.batch)

        # one-hot: inject positional encoding before node encoder  # one-hot
        if self.n_qubits_pe > 0:                                   # one-hot
            x_dense = self._inject_node_pe(x_dense)                # one-hot

        h = self.node_encoder(x_dense)

        adj_J = to_dense_adj(
            data.edge_index, data.batch, data.edge_attr
        ).squeeze(-1)
        adj_J = adj_J + adj_J.transpose(1, 2)
        e = self.edge_encoder(adj_J.unsqueeze(-1))

        for layer in self.layers:
            h, e = layer(h, e)

        # edge features, same as before
        batch_idx = data.batch[data.edge_index[0]]
        N = x_dense.size(1)
        src_local = data.edge_index[0] % N
        dst_local = data.edge_index[1] % N
        e_ij = e[batch_idx, src_local, dst_local]
        edge_input = torch.cat([e_ij, data.edge_attr], dim=-1)

        # replace output MLP by two MLPs:
        magnitude = self.magnitude_head(edge_input).squeeze(-1) # returns softmax(output) so [0,1]
        sign_logit = self.sign_head(edge_input).squeeze(-1) # return class logit (-inf, inf)

        # reconstruct ZZ for eval/logging
        sign_prob = torch.sigmoid(sign_logit)
        zz = (2.0 * sign_prob - 1.0) * magnitude

        return {
            "magnitude": magnitude,
            "sign_logit": sign_logit,
            "zz": zz,
        }


class DualHeadLoss(nn.Module):
    """
    Loss for DualHeadGNN.

    L = w_mag * MSE(pred_mag, |target|) + log_sigma_mag
      + w_sign * BCE(sign_logit, sign_target) + log_sigma_sign

    where w_x = exp(-2 * log_sigma_x)  (learned automatically)
    and BCE is weighted per-edge by |target| so near-zero
    targets don't penalize the sign head.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mag_pred,
        sign_logit,
        target,
        log_sigma_mag,
        log_sigma_sign,
    ):
        """
        Args:
            takes the mag_pred: (num_edges,) predicted |ZZ| in [0,1]
            and the sign_logit: (num_edges,) raw logit for sign
            target: (num_edges,) true ZZ in [-1,1]
            log_sigma_mag: scalar parameter from model
            log_sigma_sign: scalar parameter from model

        Returns:
            (total_loss, metrics_dict)
        """
        mag_target = target.abs()
        sign_target = (target > 0).to(target.dtype)

        # magnitude loss: MSE on |ZZ|
        mag_loss = F.mse_loss(mag_pred, mag_target)

        # sign loss: BCE weighted by |target|
        # When |target| ~ 0, sign is ambiguous -> low weight
        bce_per_edge = F.binary_cross_entropy_with_logits(
            sign_logit, sign_target, reduction='none'
        )
        weights = mag_target
        # compute relative loss
        w_sum = weights.sum().clamp(min=1e-8)
        sign_loss = (bce_per_edge * weights).sum() / w_sum

        # weigh the losses
        # L = (1/2σ²) * task_loss + log(σ)
        # = exp(-2*log_σ) * task_loss + log_σ
        w_mag = torch.exp(-2.0 * log_sigma_mag)
        w_sign = torch.exp(-2.0 * log_sigma_sign)

        total_loss = (
            w_mag * mag_loss + log_sigma_mag
            + w_sign * sign_loss + log_sigma_sign
        )

        # metrics for logging
        with torch.no_grad():
            sign_pred = (sign_logit > 0).to(target.dtype)   # if >0, true; if <0 false.
            # i'll define sign accuracy only on edges where sign is not ambiguous for now, can change later
            sig_mask = mag_target > 0.1
            if sig_mask.sum() > 0:
                sign_acc = (
                    (sign_pred[sig_mask] == sign_target[sig_mask]).float().mean().item()
                )
            else:
                sign_acc = 1.0

        metrics = {
            "mag_loss": mag_loss.item(),
            "sign_loss": sign_loss.item(),
            "sign_acc": sign_acc,
            "sigma_mag": torch.exp(log_sigma_mag).item(), #
            "sigma_sign": torch.exp(log_sigma_sign).item(),
        }

        return total_loss, metrics


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
            n_qubits_pe=cfg.model.get("n_qubits_pe", 0), # added for one hot, also in dual gnn
        )

    # dual-head: new model type
    elif model_name == "dual_head_gnn":
        return DualHeadGNN(
            node_dim=dims.get("node_dim", 1),
            edge_dim=dims.get("edge_dim", 1),
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            activation=cfg.model.activation,
            norm=cfg.model.get("norm", "layer_norm"),
            dropout=cfg.model.get("dropout", 0.0),
            pooling=cfg.model.get("pooling", "mean"),
            n_qubits_pe=cfg.model.get("n_qubits_pe", 0),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from ['mlp', 'gnn', 'dual_head_gnn']")
