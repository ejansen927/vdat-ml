"""
Dataset classes for loading VDAT data.

Two VDAT dataset types:
    - QuantumDataset: Tensor data for MLP, returns (X, y) tuples
    - QuantumGraphDataset: Graph data for GNN, returns PyG Data objects

Data files are .pt (torch.save) containing:
    - "X": ML input [Xi, Jij], shape (N, n_qubits + n_edges)
    - "y": ML target [ZZij], shape (N, n_edges)
    - "A": Algorithm input [hi, Jij, theta] (the exact input to the VDAT code)
"""

import itertools
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import hydra
from torch_geometric.loader import DataLoader as GeomDataLoader


class QuantumDataset(Dataset):
    """
    Dataset for tensor (non-graph) data.
    
    For MLP training. Returns (X, y) tuples.
    
    Args:
        root: Directory containing train.pt, val.pt, test.pt
        train_file: Training data filename
        val_file: Validation data filename
        test_file: Test data filename (optional)
        split: Which split to load ('train', 'val', 'test')
    """
    
    def __init__(
        self,
        root: str,
        train_file: str = "train.pt",
        val_file: str = "val.pt",
        test_file: Optional[str] = None,
        split: str = "train",
        **kwargs,  # Absorb extra config params
    ):
        self.root = Path(root)
        self.split = split
        
        file_map = {
            "train": train_file,
            "val": val_file,
            "test": test_file,
        }
        
        data_file = file_map.get(split)
        if data_file is None:
            raise ValueError(f"No file specified for split '{split}'")
        
        data_path = self.root / data_file
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # loading data here
        data = torch.load(data_path, weights_only=False)
        
        self.X = data["X"].double()
        self.y = data["y"].double()
        
        # ensure y has correct shape
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(-1)
        
        # store metadata if present
        self.metadata = {k: v for k, v in data.items() if k not in ["X", "y"]}
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    @property
    def input_dim(self) -> int:
        return self.X.shape[1]
    
    @property
    def output_dim(self) -> int:
        return self.y.shape[1]


class QuantumGraphDataset(Dataset):
    """
    Dataset for graph-structured data (PyTorch Geometric).
    
    For GNN training. Converts stored tensors to PyG Data objects on the fly.
    
    Data format in .pt file:
        - "X": [Xi, Jij] concatenated, shape (N, n_qubits + n_edges)
        - "y": [ZZij], shape (N, n_edges)
    
    We split X back into:
        - Xi (node features): shape (n_qubits,) per sample
        - Jij (edge features): shape (n_edges,) per sample
    
    Args:
        root: Directory containing train.pt, val.pt, test.pt
        n_qubits: Number of qubits (nodes) - required to split X correctly
        train_file: Training data filename
        val_file: Validation data filename
        test_file: Test data filename (optional)
        split: Which split to load
    """
    
    def __init__(
        self,
        root: str,
        train_file: str = "train.pt",
        val_file: str = "val.pt",
        test_file: Optional[str] = None,
        split: str = "train",
        n_qubits: Optional[int] = None,
        **kwargs,
    ):
        self.Data = Data
        self.root = Path(root)
        self.split = split
        
        file_map = {"train": train_file, "val": val_file, "test": test_file}
        data_file = file_map.get(split)
        if data_file is None:
            raise ValueError(f"No file specified for split '{split}'")
        
        data_path = self.root / data_file
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        data = torch.load(data_path, weights_only=False)
        
        self.X = data["X"].double()
        self.y = data["y"].double()
        
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(-1)
        
        # Determine n_qubits and n_edges
        if n_qubits is not None:
            self.n_qubits = n_qubits
        else:
            # Try to infer from config stored in kwargs
            self.n_qubits = kwargs.get("n_qubits")
            if self.n_qubits is None:
                raise ValueError(
                    "n_qubits must be specified in config for graph dataset. "
                    "Add 'n_qubits: N' to your data config YAML."
                )
        
        self.n_edges = len(list(itertools.combinations(range(self.n_qubits), 2)))
        
        # Verify dimensions match
        expected_input_dim = self.n_qubits + self.n_edges
        if self.X.shape[1] != expected_input_dim:
            raise ValueError(
                f"Input dimension mismatch: got {self.X.shape[1]}, "
                f"expected {expected_input_dim} (n_qubits={self.n_qubits}, n_edges={self.n_edges})"
            )
        
        # Build edge_index (fully connected graph, 0-indexed)
        edges = list(itertools.combinations(range(self.n_qubits), 2))
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Store metadata
        self.metadata = {k: v for k, v in data.items() if k not in ["X", "y"]}
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        """
        Get a single graph.
        
        Splits X[idx] = [Xi, Jij] into node and edge features.
        """
        x_full = self.X[idx]
        
        # Split into Xi (nodes) and Jij (edges)
        Xi = x_full[:self.n_qubits]
        Jij = x_full[self.n_qubits:]
        
        # Node features: (n_qubits, 1)
        node_features = Xi.unsqueeze(-1)
        
        # Edge features: (n_edges, 1)
        edge_features = Jij.unsqueeze(-1)
        
        # Target: (n_edges, 1) or (n_edges,)
        target = self.y[idx]
        
        return self.Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=edge_features,
            y=target,
        )
    
    @property
    def input_dim(self) -> int:
        return self.X.shape[1]
    
    @property
    def output_dim(self) -> int:
        return self.y.shape[1]
    
    @property
    def node_dim(self) -> int:
        return 1
    
    @property
    def edge_dim(self) -> int:
        return 1


def setup_data(cfg):
    """
    Setup data loaders from Hydra config.
    
    Main entry point called by train.py.
    
    Args:
        cfg: Hydra DictConfig with cfg.data containing dataset config
        
    Returns:
        train_loader, val_loader, test_loader (test may be None)
        dims dict with input_dim, output_dim, n_qubits, n_edges
    """
    
    is_graph = cfg.data.get("is_graph", False)
    
    # Instantiate datasets for each split
    train_dataset = hydra.utils.instantiate(cfg.data, split="train")
    val_dataset = hydra.utils.instantiate(cfg.data, split="val")
    
    # Test set is optional
    test_dataset = None
    if cfg.data.get("test_file") is not None:
        try:
            test_dataset = hydra.utils.instantiate(cfg.data, split="test")
        except FileNotFoundError:
            print("Test file specified but not found, skipping test set")
    
    # Get dimensions from dataset
    dims = {
        "input_dim": train_dataset.input_dim,
        "output_dim": train_dataset.output_dim,
    }
    
    if is_graph:
        dims["n_qubits"] = train_dataset.n_qubits
        dims["n_edges"] = train_dataset.n_edges
        dims["node_dim"] = train_dataset.node_dim
        dims["edge_dim"] = train_dataset.edge_dim
    
    # Choose appropriate DataLoader class
    if is_graph:
        LoaderClass = GeomDataLoader
    else:
        LoaderClass = DataLoader
    
    # Common loader kwargs
    loader_kwargs = {
        "num_workers": cfg.data.get("num_workers", 0),
        "pin_memory": cfg.data.get("pin_memory", False),
    }
    
    train_loader = LoaderClass(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    
    val_loader = LoaderClass(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = LoaderClass(
            test_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
    
    return train_loader, val_loader, test_loader, dims
