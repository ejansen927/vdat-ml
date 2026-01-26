#!/usr/bin/env python3
"""
Main training script with Hydra configuration and W&B logging.

Usage:
    # Basic run
    python src/train.py
    
    # Override model
    python src/train.py model=gnn data=random_4q_graph
    
    # Override hyperparameters
    python src/train.py model=mlp optimizer.lr=1e-4 training.epochs=200
    
    # Multirun sweep
    python src/train.py -m model=mlp,gnn optimizer.lr=1e-3,1e-4
    
    # Debug mode (no W&B, few epochs)
    python src/train.py experiment=debug
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for local imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import time
import logging
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

# Local imports (now sys.path includes src/)
import models
import data
import utils

from models import build_model
from data import setup_data
from utils import (
    set_seed,
    get_device,
    get_loss_fn,
    count_parameters,
    format_parameters,
    EarlyStopping,
    save_checkpoint,
    AverageMeter,
)

log = logging.getLogger(__name__)


# =============================================================================
# W&B Setup
# =============================================================================

def setup_wandb(cfg: DictConfig, dims: Dict[str, int]) -> Optional[Any]:
    """Initialize Weights & Biases logging."""
    if not cfg.logging.wandb.enabled:
        log.info("W&B logging disabled")
        return None
    
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed. Run: pip install wandb")
        return None
    
    # Convert config to dict for W&B
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Add data dimensions to config
    config_dict["dims"] = dims
    
    # Auto-generate run name: {model}_{data}_{lr}
    run_name = f"{cfg.model.name}_{cfg.data.name}_lr{cfg.optimizer.lr}"
    
    run = wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        name=run_name,
        config=config_dict,
        tags=list(cfg.project.tags) if cfg.project.tags else [],
        group=cfg.project.experiment,
        reinit=True,
    )
    
    log.info(f"W&B run initialized: {run.url}")
    return run


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_graph: bool,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    
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
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), target.size(0))
    
    return {"train_loss": loss_meter.avg}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_graph: bool,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    mse_error = 0.0
    mse_target = 0.0
    
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
        
        # For relative MSE
        mse_error += torch.sum((pred - target) ** 2).item()
        mse_target += torch.sum(target ** 2).item()
    
    # Relative MSE
    epsilon = 1e-8
    rel_loss = mse_error / (mse_target + epsilon)
    
    return {
        "val_loss": loss_meter.avg,
        "val_rel_loss": rel_loss,
    }


@torch.no_grad()
def test(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_graph: bool,
) -> Dict[str, float]:
    """Test the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    mse_error = 0.0
    mse_target = 0.0
    
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
    
    epsilon = 1e-8
    rel_loss = mse_error / (mse_target + epsilon)
    
    return {
        "test_loss": loss_meter.avg,
        "test_rel_loss": rel_loss,
    }


# =============================================================================
# Main
# =============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    
    # Print config
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    log.info(f"Using device: {device}")
    
    # Setup data
    log.info("Loading data...")
    train_loader, val_loader, test_loader, dims = setup_data(cfg)
    log.info(f"Data dimensions: {dims}")
    log.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset) if test_loader else 0}")
    
    is_graph = cfg.data.get("is_graph", False)
    
    # Build model
    model = build_model(cfg, dims).to(device)
    param_counts = count_parameters(model)
    log.info(f"Model: {cfg.model.name} | Parameters: {format_parameters(param_counts['trainable'])} trainable")
    
    # Initialize W&B
    wandb_run = setup_wandb(cfg, dims)
    
    # Log model info to W&B
    if wandb_run:
        import wandb
        wandb.log({
            "model/num_parameters": param_counts["trainable"],
            "model/num_parameters_total": param_counts["total"],
            "data/train_size": len(train_loader.dataset),
            "data/val_size": len(val_loader.dataset),
            "data/test_size": len(test_loader.dataset) if test_loader else 0,
            "training/batch_size": cfg.training.batch_size,
        })
    
    # Setup optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Setup scheduler
    scheduler = None
    if cfg.scheduler.get("_target_") is not None:
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    
    # Setup loss
    loss_name = cfg.training.get("loss", "mse")
    criterion = get_loss_fn(loss_name)
    log.info(f"Loss function: {loss_name}")
    
    # Setup early stopping
    early_stopping = None
    if cfg.training.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
        )
    
    # Checkpoint directory - use Hydra's output dir
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints: {ckpt_dir}")
    
    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    
    log.info("=" * 70)
    log.info(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Rel':<12} | {'LR':<10}")
    log.info("=" * 70)
    
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, is_graph
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, is_graph)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Update scheduler
        if scheduler is not None:
            # ReduceLROnPlateau requires val_loss, others just step()
            if "ReduceLROnPlateau" in str(cfg.scheduler.get("_target_", "")):
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()
        
        # Logging
        epoch_time = time.time() - epoch_start
        
        if epoch == 1 or epoch % cfg.logging.console.log_freq == 0:
            log.info(
                f"{epoch:<8} | {train_metrics['train_loss']:<12.6f} | "
                f"{val_metrics['val_loss']:<12.6f} | {val_metrics['val_rel_loss']:<12.6f} | "
                f"{current_lr:<10.2e}"
            )
        
        # Log to W&B
        if wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["train_loss"],
                "val/loss": val_metrics["val_loss"],
                "val/rel_loss": val_metrics["val_rel_loss"],
                "train/lr": current_lr,
                "train/epoch_time": epoch_time,
            })
        
        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            
            if cfg.checkpoint.save_best:
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model, optimizer, epoch, val_metrics["val_loss"],
                    config=OmegaConf.to_container(cfg, resolve=True),
                    scheduler=scheduler,
                )
        
        # Periodic checkpoint
        if cfg.checkpoint.save_freq and epoch % cfg.checkpoint.save_freq == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch}.pt",
                model, optimizer, epoch, val_metrics["val_loss"],
                scheduler=scheduler,
            )
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics["val_loss"]):
                log.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    # Save last checkpoint
    if cfg.checkpoint.save_last:
        save_checkpoint(
            ckpt_dir / "last.pt",
            model, optimizer, epoch, val_metrics["val_loss"],
            config=OmegaConf.to_container(cfg, resolve=True),
            scheduler=scheduler,
        )
    
    log.info("=" * 70)
    log.info(f"Training complete. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Test evaluation
    if test_loader is not None:
        log.info("Running test evaluation...")
        
        # Load best checkpoint for testing
        if cfg.checkpoint.save_best and (ckpt_dir / "best.pt").exists():
            checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        
        test_metrics = test(model, test_loader, criterion, device, is_graph)
        
        log.info(f"Test Loss: {test_metrics['test_loss']:.6f}")
        log.info(f"Test Rel Loss: {test_metrics['test_rel_loss']:.6f}")
        
        if wandb_run:
            import wandb
            wandb.log({
                "test/loss": test_metrics["test_loss"],
                "test/rel_loss": test_metrics["test_rel_loss"],
            })
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["best_epoch"] = best_epoch
            wandb.summary["test_loss"] = test_metrics["test_loss"]
            wandb.summary["test_rel_loss"] = test_metrics["test_rel_loss"]
    
    # Finish W&B
    if wandb_run:
        import wandb
        wandb.finish()
    
    log.info("Done!")
    
    return best_val_loss


if __name__ == "__main__":
    main()
