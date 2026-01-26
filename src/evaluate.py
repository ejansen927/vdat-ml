#!/usr/bin/env python3
"""
Evaluation script for trained models.

Usage:
    # Evaluate best checkpoint on test set
    python src/evaluate.py checkpoint=outputs/.../checkpoints/best.pt data=random_4q
    
    # Evaluate with different data
    python src/evaluate.py checkpoint=path/to/model.pt data=gaussian_4q
    
    # Evaluate GNN model
    python src/evaluate.py checkpoint=path/to/gnn.pt data=random_4q_graph model=gnn
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import logging
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from models import build_model
from data import setup_data
from utils import (
    set_seed,
    get_device,
    get_loss_fn,
    count_parameters,
    format_parameters,
    load_checkpoint,
    AverageMeter,
)

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_graph: bool,
    split_name: str = "test",
) -> Dict[str, float]:
    """
    Evaluate model on a data split.
    
    Returns:
        Dict with loss and relative loss metrics
    """
    model.eval()
    
    loss_meter = AverageMeter()
    mse_error = 0.0
    mse_target = 0.0
    
    all_preds = []
    all_targets = []
    
    for batch in loader:
        if is_graph:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y
        else:
            x, target = batch
            x, target = x.to(device), target.to(device)
            pred = model(x)
        
        loss = criterion(pred, target)
        loss_meter.update(loss.item(), target.size(0))
        
        mse_error += torch.sum((pred - target) ** 2).item()
        mse_target += torch.sum(target ** 2).item()
        
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())
    
    epsilon = 1e-8
    rel_loss = mse_error / (mse_target + epsilon)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute additional metrics
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    max_error = torch.max(torch.abs(all_preds - all_targets)).item()
    
    return {
        f"{split_name}_loss": loss_meter.avg,
        f"{split_name}_rel_loss": rel_loss,
        f"{split_name}_mae": mae,
        f"{split_name}_max_error": max_error,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    
    # Check checkpoint is specified
    if not cfg.get("checkpoint"):
        raise ValueError("Must specify checkpoint path: python src/evaluate.py checkpoint=path/to/model.pt")
    
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    log.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Setup
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    log.info(f"Using device: {device}")
    
    # Setup data
    log.info("Loading data...")
    train_loader, val_loader, test_loader, dims = setup_data(cfg)
    log.info(f"Data dimensions: {dims}")
    
    is_graph = cfg.data.get("is_graph", False)
    
    # Build model
    model = build_model(cfg, dims).to(device)
    param_counts = count_parameters(model)
    log.info(f"Model: {cfg.model.name} | Parameters: {format_parameters(param_counts['trainable'])}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    log.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    log.info(f"Checkpoint val_loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # Setup loss
    loss_name = cfg.training.get("loss", "mse")
    criterion = get_loss_fn(loss_name)
    
    # Evaluate on all splits
    log.info("=" * 60)
    
    # Validation set
    log.info("Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion, device, is_graph, "val")
    for k, v in val_metrics.items():
        log.info(f"  {k}: {v:.6f}")
    
    # Test set
    if test_loader is not None:
        log.info("Evaluating on test set...")
        test_metrics = evaluate(model, test_loader, criterion, device, is_graph, "test")
        for k, v in test_metrics.items():
            log.info(f"  {k}: {v:.6f}")
    
    # Train set (optional, for overfitting check)
    log.info("Evaluating on train set...")
    train_metrics = evaluate(model, train_loader, criterion, device, is_graph, "train")
    for k, v in train_metrics.items():
        log.info(f"  {k}: {v:.6f}")
    
    log.info("=" * 60)
    log.info("Evaluation complete!")
    
    # Summary
    log.info("\nSummary:")
    log.info(f"  Train Loss:     {train_metrics['train_loss']:.6f}")
    log.info(f"  Val Loss:       {val_metrics['val_loss']:.6f}")
    if test_loader is not None:
        log.info(f"  Test Loss:      {test_metrics['test_loss']:.6f}")
        log.info(f"  Test Rel Loss:  {test_metrics['test_rel_loss']:.6f}")


if __name__ == "__main__":
    main()
