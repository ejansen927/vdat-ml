"""
Utility functions for training.

"""

import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device, with fallback."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        if device_str in ("cuda", "mps"):
            print(f"{device_str} not available, falling back to CPU")
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dict with 'total', 'trainable', 'non_trainable' counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def format_parameters(num: int) -> str:
    """Format parameter count for display (e.g., 1.2M, 350K)."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping handler.
    
    Tracks validation loss and signals when to stop training.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_loss
            return False
        
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
):
    """
    Save training checkpoint.
    
    Args:
        path: Where to save
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss at this checkpoint
        config: Optional config dict for reproducibility
        scheduler: Optional scheduler state
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    
    if config is not None:
        checkpoint["config"] = config
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load to
        
    Returns:
        Checkpoint dict with epoch, val_loss, etc.
    """
    checkpoint = torch.load(path, map_location=device or "cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def get_loss_fn(name: str) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        name: 'mse' or 'mae'
        
    Returns:
        Loss function module
    """
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        #"l1": nn.L1Loss,
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    
    return losses[name.lower()]()
