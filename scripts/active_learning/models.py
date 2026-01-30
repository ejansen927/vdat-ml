#!/usr/bin/env python3
"""
Surrogate Models for Active Learning.

Provides:
    - SurrogateGP: Sparse Variational GP (original)
    - SurrogateEnsemble: Deep ensemble of GNNs for uncertainty via disagreement

All surrogates implement the same interface:
    - fit(X, y, verbose=True)
    - predict(X) -> (mean, std)
    - gradients(X) -> (grad_mean, grad_std)  [for UCB']
"""

import itertools
import torch
import torch.nn as nn
import gpytorch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


# =============================================================================
# SVGP Surrogate (Original)
# =============================================================================

class SVGPModel(gpytorch.models.ApproximateGP):
    """Sparse Variational GP with RBF kernel and multi-task output."""
    
    def __init__(self, inducing_points, num_tasks=6):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_strategy, num_tasks=num_tasks
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SurrogateGP:
    """
    Wrapper for SVGP training and inference.
    
    Args:
        num_inducing: Number of inducing points (more = better fit, slower)
        num_epochs: Training epochs
        num_tasks: Output dimensions (6 for ZZij)
    """
    
    def __init__(self, num_inducing=500, num_epochs=50, num_tasks=6):
        self.num_inducing = num_inducing
        self.num_epochs = num_epochs
        self.num_tasks = num_tasks
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.likelihood = None
    
    def fit(self, X, y, verbose=True):
        """Train SVGP on labeled data."""
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        
        # Inducing points via K-means
        n_inducing = min(self.num_inducing, len(X))
        kmeans = KMeans(n_clusters=n_inducing, n_init=10, random_state=0)
        kmeans.fit(X.numpy())
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        # Initialize model
        self.model = SVGPModel(inducing_points.to(self.device), self.num_tasks).to(self.device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)
        
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)
        
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(y))
        loader = DataLoader(TensorDataset(X, y), batch_size=1024, shuffle=True)
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = -mll(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1:3d}/{self.num_epochs} | Loss: {epoch_loss/len(loader):.4f}")
        
        self.model.eval()
        self.likelihood.eval()
    
    def predict(self, X, batch_size=8192):
        """Get mean and std predictions."""
        X = torch.as_tensor(X, dtype=torch.float32)
        means, stds = [], []
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(self.device)
                pred = self.likelihood(self.model(X_batch))
                means.append(pred.mean.cpu())
                stds.append(pred.variance.sqrt().cpu())
        
        return torch.cat(means), torch.cat(stds)
    
    def gradients(self, X, batch_size=2048):
        """Get gradients of mean and std w.r.t. inputs (for UCB')."""
        X = torch.as_tensor(X, dtype=torch.float32)
        grad_means, grad_stds = [], []
        
        with gpytorch.settings.fast_pred_var():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(self.device).requires_grad_(True)
                pred = self.likelihood(self.model(X_batch))
                
                grad_mean = torch.autograd.grad(pred.mean.sum(), X_batch, retain_graph=True)[0]
                grad_std = torch.autograd.grad(pred.variance.sqrt().sum(), X_batch)[0]
                
                grad_means.append(grad_mean.detach().cpu())
                grad_stds.append(grad_std.detach().cpu())
        
        return torch.cat(grad_means), torch.cat(grad_stds)


# =============================================================================
# GNN Components (Flat Input Version for AL)
# =============================================================================

class InteractiveNetFlat(nn.Module):
    """
    Message passing layer that works with dense batched tensors.
    Matches the architecture in src/models.py but for flat input.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Message MLP: [h_src, h_dst, J_ij] -> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Update MLP: [h_i, aggregated_messages] -> h_i'
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
    
    def forward(self, h: torch.Tensor, adj_J: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node embeddings (B, N, D)
            adj_J: Adjacency with J values (B, N, N)
        Returns:
            Updated node embeddings (B, N, D)
        """
        B, N, D = h.shape
        
        # Expand for pairwise operations
        h_source = h.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)
        h_target = h.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        J_exp = adj_J.unsqueeze(-1)                   # (B, N, N, 1)
        
        # Compute messages
        msg_input = torch.cat([h_target, h_source, J_exp], dim=-1)
        messages = self.msg_mlp(msg_input)
        
        # Mask self-messages
        mask = (1 - torch.eye(N, device=h.device)).view(1, N, N, 1)
        messages = messages * mask
        
        # Aggregate (sum over source nodes)
        agg_msgs = messages.sum(dim=2)  # (B, N, D)
        
        # Update with residual
        update_input = torch.cat([h, agg_msgs], dim=-1)
        return h + self.update_mlp(update_input)


class FlatGNN(nn.Module):
    """
    GNN that takes flat tensor input [Xi, J_ij] and outputs ZZ_ij predictions.
    
    Internally reconstructs graph structure for message passing.
    Architecture matches src/models.py GNN exactly.
    
    Args:
        n_qubits: Number of qubits (nodes)
        hidden_dim: Hidden dimension
        num_layers: Number of message passing layers
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_edges = n_qubits * (n_qubits - 1) // 2
        self.hidden_dim = hidden_dim
        
        # Precompute edge indices (upper triangle)
        self.edge_pairs = list(itertools.combinations(range(n_qubits), 2))
        
        # Node encoder: Xi -> hidden
        self.node_encoder = nn.Linear(1, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList([
            InteractiveNetFlat(hidden_dim) for _ in range(num_layers)
        ])
        
        # Edge decoder: [h_src, h_dst, J_ij] -> ZZ_ij
        # Matches src/models.py exactly (NO LayerNorm in decoder)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Bounded output [-1, 1]
        )
    
    def _build_adjacency(self, J_ij: torch.Tensor) -> torch.Tensor:
        """
        Build symmetric adjacency matrix from J_ij vector.
        
        Args:
            J_ij: Edge features (B, n_edges)
        Returns:
            Symmetric adjacency (B, N, N)
        """
        B = J_ij.size(0)
        N = self.n_qubits
        device = J_ij.device
        
        adj = torch.zeros(B, N, N, device=device)
        
        for idx, (i, j) in enumerate(self.edge_pairs):
            adj[:, i, j] = J_ij[:, idx]
            adj[:, j, i] = J_ij[:, idx]  # Symmetric
        
        return adj
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            X: Input tensor (B, n_qubits + n_edges) = [Xi, J_ij]
        Returns:
            Predictions (B, n_edges)
        """
        B = X.size(0)
        
        # Split input
        Xi = X[:, :self.n_qubits]      # (B, n_qubits)
        J_ij = X[:, self.n_qubits:]    # (B, n_edges)
        
        # Encode nodes
        h = self.node_encoder(Xi.unsqueeze(-1))  # (B, N, hidden_dim)
        
        # Build adjacency
        adj_J = self._build_adjacency(J_ij)  # (B, N, N)
        
        # Message passing
        for layer in self.layers:
            h = layer(h, adj_J)
        
        # Decode edges
        outputs = []
        for idx, (i, j) in enumerate(self.edge_pairs):
            h_i = h[:, i, :]  # (B, D)
            h_j = h[:, j, :]  # (B, D)
            j_val = J_ij[:, idx:idx+1]  # (B, 1)
            
            edge_input = torch.cat([h_i, h_j, j_val], dim=-1)
            out = self.edge_decoder(edge_input)
            outputs.append(out)
        
        return torch.cat(outputs, dim=-1)  # (B, n_edges)


# =============================================================================
# Deep Ensemble Surrogate
# =============================================================================

class SurrogateEnsemble:
    """
    Deep Ensemble surrogate for Active Learning.
    
    Uses M GNNs with different random initializations.
    Uncertainty = disagreement (std across predictions).
    
    The "GNN-Committee":
        - All M GNNs have identical architecture
        - Different only in random weight initialization
        - For input x: get M predictions {y_1, ..., y_M}
        - Mean: μ(x) = (1/M) Σ y_i  (best guess)
        - Std:  σ(x) = std({y_i})    (disagreement = uncertainty)
    
    Args:
        n_members: Number of ensemble members (default: 5)
        hidden_dim: Hidden dimension for GNNs
        num_layers: Number of message passing layers
        num_epochs: Training epochs per member
        batch_size: Training batch size
        lr: Learning rate
        n_qubits: Number of qubits
        num_tasks: Output dimensions (n_edges, for compatibility)
    """
    
    def __init__(
        self,
        n_members: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        n_qubits: int = 4,
        num_tasks: int = 6,  # For compatibility with SurrogateGP interface
    ):
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_qubits = n_qubits
        self.num_tasks = num_tasks  # = n_edges
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = []
    
    def _create_model(self, seed: int) -> nn.Module:
        """Create a single GNN with given random seed."""
        torch.manual_seed(seed)
        return FlatGNN(
            n_qubits=self.n_qubits,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
    
    def fit(self, X, y, verbose=True):
        """
        Train ensemble on labeled data.
        
        Each member trained independently with different initialization.
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        
        self.models = []
        
        for m in range(self.n_members):
            if verbose:
                print(f"      Training ensemble member {m+1}/{self.n_members}...")
            
            # Create model with unique seed
            model = self._create_model(seed=42 + m * 1000).to(self.device)
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
            criterion = nn.MSELoss()
            
            loader = DataLoader(
                TensorDataset(X, y),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=len(X) > self.batch_size,  # Only drop if we have enough data
            )
            
            model.train()
            for epoch in range(self.num_epochs):
                epoch_loss = 0
                n_batches = 0
                
                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                scheduler.step()
                
                if verbose and (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    print(f"        Epoch {epoch+1:3d}/{self.num_epochs} | Loss: {avg_loss:.6f}")
            
            model.eval()
            self.models.append(model)
        
        if verbose:
            print(f"      Ensemble training complete ({self.n_members} members)")
    
    def predict(self, X, batch_size=4096):
        """
        Get mean and std predictions from ensemble.
        
        Args:
            X: Input tensor (N, input_dim)
        Returns:
            mean: (N, num_tasks) - average prediction across ensemble
            std: (N, num_tasks) - std across ensemble (disagreement = uncertainty)
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        
        all_preds = []
        
        with torch.no_grad():
            for model in self.models:
                model_preds = []
                
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i+batch_size].to(self.device)
                    pred = model(X_batch).cpu()
                    model_preds.append(pred)
                
                all_preds.append(torch.cat(model_preds, dim=0))
        
        # Stack: (n_members, N, num_tasks)
        stacked = torch.stack(all_preds, dim=0)
        
        # Mean and std across ensemble members
        mean = stacked.mean(dim=0)  # (N, num_tasks)
        std = stacked.std(dim=0)    # (N, num_tasks)
        
        return mean, std
    
    def gradients(self, X, batch_size=2048):
        """
        Get gradients of mean and std w.r.t. inputs (for UCB').
        
        Computes gradients of ensemble mean and std.
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        
        grad_means_list = []
        grad_stds_list = []
        
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(self.device).requires_grad_(True)
            
            # Get predictions from all members
            preds = []
            for model in self.models:
                pred = model(X_batch)
                preds.append(pred)
            
            stacked = torch.stack(preds, dim=0)  # (M, B, num_tasks)
            
            # Compute mean and std
            mean = stacked.mean(dim=0)  # (B, num_tasks)
            std = stacked.std(dim=0)    # (B, num_tasks)
            
            # Compute gradients
            grad_mean = torch.autograd.grad(
                mean.sum(), X_batch, retain_graph=True, create_graph=False
            )[0]
            grad_std = torch.autograd.grad(
                std.sum(), X_batch, create_graph=False
            )[0]
            
            grad_means_list.append(grad_mean.detach().cpu())
            grad_stds_list.append(grad_std.detach().cpu())
        
        return torch.cat(grad_means_list), torch.cat(grad_stds_list)


# =============================================================================
# Factory function
# =============================================================================

def create_surrogate(surrogate_type: str = 'gp', **kwargs):
    """
    Create a surrogate model.
    
    Args:
        surrogate_type: 'gp', 'ensemble', 'svm', or 'error_predictor'
        **kwargs: Arguments passed to the surrogate constructor
    
    Returns:
        Surrogate instance with fit/predict/gradients methods
    """
    surrogate_type = surrogate_type.lower()
    
    if surrogate_type == 'gp':
        return SurrogateGP(
            num_inducing=kwargs.get('num_inducing', 500),
            num_epochs=kwargs.get('num_epochs', 50),
            num_tasks=kwargs.get('num_tasks', 6),
        )
    
    elif surrogate_type == 'ensemble':
        return SurrogateEnsemble(
            n_members=kwargs.get('n_members', 5),
            hidden_dim=kwargs.get('hidden_dim', 64),
            num_layers=kwargs.get('num_layers', 3),
            num_epochs=kwargs.get('num_epochs', 30),
            batch_size=kwargs.get('batch_size', 256),
            lr=kwargs.get('lr', 1e-3),
            n_qubits=kwargs.get('n_qubits', 4),
            num_tasks=kwargs.get('num_tasks', 6),
        )
    
    elif surrogate_type == 'svm':
        return SurrogateSVM(
            n_members=kwargs.get('n_members', 5),
            C=kwargs.get('svm_C', 1.0),
            epsilon=kwargs.get('svm_epsilon', 0.1),
            kernel=kwargs.get('svm_kernel', 'rbf'),
            gamma=kwargs.get('svm_gamma', 'scale'),
            max_samples=kwargs.get('svm_max_samples', 50000),
            num_tasks=kwargs.get('num_tasks', 6),
        )
    
    elif surrogate_type == 'error_predictor':
        predictor_path = kwargs.get('error_predictor_path')
        if predictor_path is None:
            raise ValueError(
                "Error predictor surrogate requires --error_predictor_path argument. "
                "First run analyze_errors.py to train an error predictor."
            )
        return SurrogateErrorPredictor(
            predictor_path=predictor_path,
            n_qubits=kwargs.get('n_qubits', 4),
        )
    
    else:
        raise ValueError(
            f"Unknown surrogate_type: {surrogate_type}. "
            f"Choose from ['gp', 'ensemble', 'svm', 'error_predictor']"
        )


# =============================================================================
# SVM Ensemble Surrogate
# =============================================================================

class SurrogateSVM:
    """
    SVM Ensemble surrogate for Active Learning.
    
    Uses an ensemble of SVR (Support Vector Regression) models for
    uncertainty estimation via disagreement.
    
    Since SVR doesn't provide native uncertainty, we use:
    - Bootstrap sampling: each SVR trained on different subsample
    - Disagreement (std across predictions) as uncertainty
    
    Note: Does NOT support gradients() - use 'uncertainty' acquisition only.
    
    Args:
        n_members: Number of SVR ensemble members (default: 5)
        C: Regularization parameter (default: 1.0)
        epsilon: Epsilon in epsilon-SVR model (default: 0.1)
        kernel: Kernel type ('rbf', 'linear', 'poly') (default: 'rbf')
        gamma: Kernel coefficient (default: 'scale')
        max_samples: Max samples per SVR (subsample if larger) (default: 50000)
        num_tasks: Output dimensions (default: 6)
    """
    
    def __init__(
        self,
        n_members: int = 5,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        max_samples: int = 50000,
        num_tasks: int = 6,
    ):
        self.n_members = n_members
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.max_samples = max_samples
        self.num_tasks = num_tasks
        
        self.models = []  # List of lists: [member][task] = SVR
    
    def fit(self, X, y, verbose=True):
        """
        Train ensemble of SVRs on labeled data.
        
        Each member is trained on a bootstrap sample.
        Each output dimension gets its own SVR.
        """
        from sklearn.svm import SVR
        import numpy as np
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = len(X)
        
        self.models = []
        
        for m in range(self.n_members):
            if verbose:
                print(f"      Training SVM ensemble member {m+1}/{self.n_members}...")
            
            # Bootstrap sample (with replacement)
            rng = np.random.default_rng(seed=42 + m * 1000)
            
            # Subsample if too large (SVR scales poorly with n_samples)
            sample_size = min(n_samples, self.max_samples)
            idx = rng.choice(n_samples, size=sample_size, replace=True)
            
            X_sample = X[idx]
            y_sample = y[idx]
            
            # Train one SVR per output dimension
            member_models = []
            for t in range(self.num_tasks):
                svr = SVR(
                    C=self.C,
                    epsilon=self.epsilon,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    cache_size=1000,  # MB, helps with large datasets
                )
                svr.fit(X_sample, y_sample[:, t])
                member_models.append(svr)
                
                if verbose:
                    print(f"        Task {t+1}/{self.num_tasks} done "
                          f"({len(svr.support_vectors_)} support vectors)")
            
            self.models.append(member_models)
        
        if verbose:
            print(f"      SVM ensemble training complete ({self.n_members} members)")
    
    def predict(self, X, batch_size=10000):
        """
        Get mean and std predictions from SVM ensemble.
        
        Args:
            X: Input array (N, input_dim)
        Returns:
            mean: (N, num_tasks) - average prediction across ensemble
            std: (N, num_tasks) - std across ensemble (disagreement = uncertainty)
        """
        import numpy as np
        
        X = np.asarray(X)
        n_samples = len(X)
        
        # Collect predictions from all members
        all_preds = []
        
        for member_models in self.models:
            member_pred = np.zeros((n_samples, self.num_tasks))
            
            # Predict in batches (SVR predict is fast but memory can be issue)
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                for t, svr in enumerate(member_models):
                    member_pred[i:i+batch_size, t] = svr.predict(X_batch)
            
            all_preds.append(member_pred)
        
        # Stack: (n_members, N, num_tasks)
        stacked = np.stack(all_preds, axis=0)
        
        # Mean and std across ensemble members
        mean = stacked.mean(axis=0)  # (N, num_tasks)
        std = stacked.std(axis=0)    # (N, num_tasks)
        
        # Convert to torch tensors for compatibility with acquisition functions
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
    
    def gradients(self, X, batch_size=2048):
        """
        SVM does not support efficient gradients.
        
        Raises error - use 'uncertainty' acquisition instead of 'ucb_prime'.
        """
        raise NotImplementedError(
            "SVM surrogate does not support gradients. "
            "Use --acquisition uncertainty instead of ucb_prime."
        )


# =============================================================================
# Error Predictor Surrogate
# =============================================================================

class ErrorPredictorNet(nn.Module):
    """
    Simple MLP that predicts GNN error from Hamiltonian parameters.
    
    Input: [h_i, J_ij, θ] or [h_i, J_ij] (n_qubits + n_edges + 1 or n_qubits + n_edges)
    Output: predicted error (scalar)
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


class SurrogateErrorPredictor:
    """
    Error Predictor surrogate for Active Learning.
    
    Uses a pre-trained error predictor that maps Hamiltonian parameters
    [h_i, J_ij] → expected GNN error.
    
    This targets where the model is WRONG, not just UNCERTAIN.
    
    Note: 
    - predict() returns (zeros, predicted_error) to match surrogate interface
    - The "std" output is actually predicted error, used for uncertainty acquisition
    - Does NOT support gradients - use 'uncertainty' acquisition only
    
    Args:
        predictor_path: Path to saved error predictor checkpoint
        n_qubits: Number of qubits (to handle theta padding)
    """
    
    def __init__(self, predictor_path: str, n_qubits: int = 4):
        self.predictor_path = predictor_path
        self.n_qubits = n_qubits
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load error predictor
        checkpoint = torch.load(predictor_path, map_location=self.device, weights_only=False)
        
        input_dim = checkpoint['input_dim']
        self.model = ErrorPredictorNet(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.input_dim = input_dim
        self.needs_theta = (input_dim == n_qubits + n_qubits * (n_qubits - 1) // 2 + 1)
    
    def fit(self, X, y, verbose=True):
        """
        No-op: Error predictor is pre-trained.
        
        Kept for interface compatibility with other surrogates.
        """
        if verbose:
            print(f"      Error predictor loaded from: {self.predictor_path}")
            print(f"      (No training needed - using pre-trained predictor)")
    
    def predict(self, X, batch_size=8192):
        """
        Predict error for candidates.
        
        Args:
            X: Candidates [h_i, J_ij], shape (N, n_qubits + n_edges)
        
        Returns:
            mean: zeros (placeholder for interface compatibility)
            std: predicted errors (higher = model likely wrong here)
        """
        import numpy as np
        
        X = torch.as_tensor(X, dtype=torch.float32)
        n_samples = len(X)
        
        # If predictor expects theta, add random theta
        if self.needs_theta:
            theta = torch.rand(n_samples, 1) * (np.pi / 2)
            X = torch.cat([X, theta], dim=-1)
        
        all_errors = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size].to(self.device)
                pred_error = self.model(X_batch).cpu()
                all_errors.append(pred_error)
        
        errors = torch.cat(all_errors)
        
        # Return (mean=0, std=error) so acquisition functions use error as "uncertainty"
        # This way uncertainty acquisition will select high-error candidates
        mean = torch.zeros(n_samples, 6)  # Placeholder, not used
        std = errors.unsqueeze(-1).expand(-1, 6)  # Broadcast to match output shape
        
        return mean, std
    
    def gradients(self, X, batch_size=2048):
        """
        Error predictor does not support gradients.
        
        Use --acquisition uncertainty instead of ucb_prime.
        """
        raise NotImplementedError(
            "Error predictor surrogate does not support gradients. "
            "Use --acquisition uncertainty instead of ucb_prime."
        )
