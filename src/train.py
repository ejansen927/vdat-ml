#!/usr/bin/env python3
"""
Main training script with Hydra configuration and W&B logging.

Usage:
    python src/train.py model=gnn data=random_4q_graph
    python src/train.py model=mlp optimizer.lr=1e-4 training.epochs=200
    python src/train.py -m model=mlp,gnn optimizer.lr=1e-3,1e-4
    python src/train.py model.name=dual_head_gnn data=random_4q_graph
"""

import sys
import os
from pathlib import Path

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

import models
import data
import utils

from models import build_model
from models import DualHeadLoss
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

torch.set_default_dtype(torch.float64)


def build_run_name(cfg: DictConfig) -> str:
    if cfg.get("run_name"):
        return cfg.run_name
    
    default_fields = ["model", "data", "layers", "hidden", "batch"]
    fields = list(cfg.get("run_name_fields", default_fields))
    
    parts = []
    for field in fields:
        if field == "model":
            parts.append(cfg.model.name.upper())
        elif field == "data":
            name = cfg.data.name.replace("_graph", "").replace("_", "")
            parts.append(name)
        elif field == "lr":
            parts.append(f"LR{cfg.optimizer.lr}")
        elif field == "batch":
            parts.append(f"B{cfg.training.batch_size}")
        elif field == "epochs":
            parts.append(f"E{cfg.training.epochs}")
        elif field == "layers":
            if cfg.model.name in ("gnn", "dual_head_gnn"):
                parts.append(f"{cfg.model.num_layers}L")
        elif field == "hidden":
            if cfg.model.name in ("gnn", "dual_head_gnn"):
                parts.append(f"H{cfg.model.hidden_dim}")
            elif cfg.model.name == "mlp":
                dims = cfg.model.hidden_dims
                if all(d == dims[0] for d in dims):
                    parts.append(f"H{dims[0]}x{len(dims)}")
                else:
                    parts.append(f"H{dims}")
        elif field == "arch":
            if cfg.model.name in ("gnn", "dual_head_gnn"):
                parts.append(f"{cfg.model.num_layers}L_H{cfg.model.hidden_dim}")
            elif cfg.model.name == "mlp":
                parts.append(f"{cfg.model.hidden_dims}")
    
    return "_".join(parts)


def setup_wandb(cfg: DictConfig, dims: Dict[str, int]) -> Optional[Any]:
    if not cfg.logging.wandb.enabled:
        log.info("W&B logging disabled")
        return None
    
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed. Run: pip install wandb")
        return None
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["dims"] = dims
    run_name = build_run_name(cfg)
    
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


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_graph: bool,
) -> Dict[str, float]:
    model.train()
    
    loss_meter = AverageMeter()
    mag_loss_meter = AverageMeter()
    sign_loss_meter = AverageMeter()
    sign_acc_meter = AverageMeter()
    zz_mse_sum = 0.0
    zz_count = 0
    
    for batch in loader:
        if is_graph:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y
        else:
            x, target = batch
            x, target = x.to(device), target.to(device)
            output = model(x)
        
        if isinstance(output, dict):
            loss, metrics = criterion(
                output["magnitude"],
                output["sign_logit"],
                target,
                model.log_sigma_mag,
                model.log_sigma_sign,
            )
            mag_loss_meter.update(metrics["mag_loss"], target.size(0))
            sign_loss_meter.update(metrics["sign_loss"], target.size(0))
            sign_acc_meter.update(metrics["sign_acc"], target.size(0))
            with torch.no_grad():
                zz_mse_sum += torch.sum((output["zz"] - target) ** 2).item()
                zz_count += target.numel()
        else:
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), target.size(0))
    
    result = {"train_loss": loss_meter.avg}
    if mag_loss_meter.count > 0:
        result["train_mag_loss"] = mag_loss_meter.avg
        result["train_sign_loss"] = sign_loss_meter.avg
        result["train_sign_acc"] = sign_acc_meter.avg
        result["train_zz_mse"] = zz_mse_sum / max(zz_count, 1)
    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_graph: bool,
) -> Dict[str, float]:
    model.eval()
    
    loss_meter = AverageMeter()
    mse_error = 0.0
    mse_target = 0.0
    mag_loss_meter = AverageMeter()
    sign_loss_meter = AverageMeter()
    sign_acc_meter = AverageMeter()
    zz_mse_sum = 0.0
    zz_count = 0
    
    for batch in loader:
        if is_graph:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y
        else:
            x, target = batch
            x, target = x.to(device), target.to(device)
            output = model(x)
        
        if isinstance(output, dict):
            loss, metrics = criterion(
                output["magnitude"],
                output["sign_logit"],
                target,
                model.log_sigma_mag,
                model.log_sigma_sign,
            )
            mag_loss_meter.update(metrics["mag_loss"], target.size(0))
            sign_loss_meter.update(metrics["sign_loss"], target.size(0))
            sign_acc_meter.update(metrics["sign_acc"], target.size(0))
            pred = output["zz"]
            zz_mse_sum += torch.sum((pred - target) ** 2).item()
            zz_count += target.numel()
        else:
            pred = output
            loss = criterion(pred, target)
        
        loss_meter.update(loss.item(), target.size(0))
        mse_error += torch.sum((pred - target) ** 2).item()
        mse_target += torch.sum(target ** 2).item()
    
    epsilon = 1e-8
    rel_loss = mse_error / (mse_target + epsilon)
    
    result = {
        "val_loss": loss_meter.avg,
        "val_rel_loss": rel_loss,
    }
    if mag_loss_meter.count > 0:
        result["val_mag_loss"] = mag_loss_meter.avg
        result["val_sign_loss"] = sign_loss_meter.avg
        result["val_sign_acc"] = sign_acc_meter.avg
        result["val_zz_mse"] = zz_mse_sum / max(zz_count, 1)
    return result


@torch.no_grad()
def test(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_graph: bool,
) -> Dict[str, float]:
    model.eval()
    
    loss_meter = AverageMeter()
    mse_error = 0.0
    mse_target = 0.0
    mag_loss_meter = AverageMeter()
    sign_loss_meter = AverageMeter()
    sign_acc_meter = AverageMeter()
    zz_mse_sum = 0.0
    zz_count = 0
    
    for batch in loader:
        if is_graph:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y
        else:
            x, target = batch
            x, target = x.to(device), target.to(device)
            output = model(x)
        
        if isinstance(output, dict):
            loss, metrics = criterion(
                output["magnitude"],
                output["sign_logit"],
                target,
                model.log_sigma_mag,
                model.log_sigma_sign,
            )
            mag_loss_meter.update(metrics["mag_loss"], target.size(0))
            sign_loss_meter.update(metrics["sign_loss"], target.size(0))
            sign_acc_meter.update(metrics["sign_acc"], target.size(0))
            pred = output["zz"]
            zz_mse_sum += torch.sum((pred - target) ** 2).item()
            zz_count += target.numel()
        else:
            pred = output
            loss = criterion(pred, target)
        
        loss_meter.update(loss.item(), target.size(0))
        mse_error += torch.sum((pred - target) ** 2).item()
        mse_target += torch.sum(target ** 2).item()
    
    epsilon = 1e-8
    rel_loss = mse_error / (mse_target + epsilon)
    
    result = {
        "test_loss": loss_meter.avg,
        "test_rel_loss": rel_loss,
    }
    if mag_loss_meter.count > 0:
        result["test_mag_loss"] = mag_loss_meter.avg
        result["test_sign_loss"] = sign_loss_meter.avg
        result["test_sign_acc"] = sign_acc_meter.avg
        result["test_zz_mse"] = zz_mse_sum / max(zz_count, 1)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    log.info(f"Using device: {device}")
    
    log.info("Loading data...")
    train_loader, val_loader, test_loader, dims = setup_data(cfg)
    log.info(f"Data dimensions: {dims}")
    log.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset) if test_loader else 0}")
    
    is_graph = cfg.data.get("is_graph", False)
    is_dual_head = cfg.model.name.lower() == "dual_head_gnn"
    
    model = build_model(cfg, dims).double().to(device)
    param_counts = count_parameters(model)
    log.info(f"Model: {cfg.model.name} | Parameters: {format_parameters(param_counts['trainable'])} trainable")
    
    wandb_run = setup_wandb(cfg, dims)
    
    if wandb_run:
        import wandb
        wandb.log({
            "model/num_parameters": param_counts["trainable"],
            "data/train_size": len(train_loader.dataset),
            "data/val_size": len(val_loader.dataset),
        })
    
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    scheduler = None
    if cfg.scheduler.get("_target_") is not None:
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    
    if is_dual_head:
        criterion = DualHeadLoss().to(device)
        log.info("Loss: DualHeadLoss (mag MSE + weighted BCE + uncertainty)")
    else:
        loss_name = cfg.training.get("loss", "mse")
        criterion = get_loss_fn(loss_name)
        log.info(f"Loss: {loss_name}")
    
    early_stopping = None
    if cfg.training.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
        )
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints: {ckpt_dir}")
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    if is_dual_head:
        log.info("=" * 95)
        log.info(
            f"{'Ep':<5} | {'ZZ MSE (t)':>10} | {'ZZ MSE (v)':>10} | "
            f"{'Mag (v)':>10} | {'Sign (v)':>10} | "
            f"{'Acc (v)':>8} | {'LR':>9}"
        )
        log.info("-" * 95)
    else:
        log.info("=" * 70)
        log.info(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Rel':<12} | {'LR':<10}")
        log.info("-" * 70)
    
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.time()
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, is_graph)
        val_metrics = validate(model, val_loader, criterion, device, is_graph)
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        if scheduler is not None:
            if "ReduceLROnPlateau" in str(cfg.scheduler.get("_target_", "")):
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        if epoch == 1 or epoch % cfg.logging.console.log_freq == 0:
            if is_dual_head:
                log.info(
                    f"{epoch:<5} | "
                    f"{train_metrics.get('train_zz_mse', 0):>10.2e} | "
                    f"{val_metrics.get('val_zz_mse', 0):>10.2e} | "
                    f"{val_metrics.get('val_mag_loss', 0):>10.2e} | "
                    f"{val_metrics.get('val_sign_loss', 0):>10.4f} | "
                    f"{val_metrics.get('val_sign_acc', 0):>8.4f} | "
                    f"{current_lr:>9.2e}"
                )
            else:
                log.info(
                    f"{epoch:<8} | {train_metrics['train_loss']:<12.6f} | "
                    f"{val_metrics['val_loss']:<12.6f} | {val_metrics['val_rel_loss']:<12.6f} | "
                    f"{current_lr:<10.2e}"
                )
        
        if wandb_run:
            import wandb
            wandb_log = {
                "epoch": epoch,
                "train/loss": train_metrics["train_loss"],
                "val/loss": val_metrics["val_loss"],
                "val/rel_loss": val_metrics["val_rel_loss"],
                "train/lr": current_lr,
                "train/epoch_time": epoch_time,
            }
            if "train_zz_mse" in train_metrics:
                wandb_log.update({
                    "train/zz_mse": train_metrics["train_zz_mse"],
                    "train/mag_loss": train_metrics["train_mag_loss"],
                    "train/sign_loss": train_metrics["train_sign_loss"],
                    "train/sign_acc": train_metrics["train_sign_acc"],
                })
            if "val_zz_mse" in val_metrics:
                wandb_log.update({
                    "val/zz_mse": val_metrics["val_zz_mse"],
                    "val/mag_loss": val_metrics["val_mag_loss"],
                    "val/sign_loss": val_metrics["val_sign_loss"],
                    "val/sign_acc": val_metrics["val_sign_acc"],
                })
            if hasattr(model, 'log_sigma_mag'):
                wandb_log["train/sigma_mag"] = torch.exp(model.log_sigma_mag).item()
                wandb_log["train/sigma_sign"] = torch.exp(model.log_sigma_sign).item()
            wandb.log(wandb_log)
        
        val_metric_for_best = val_metrics.get("val_zz_mse", val_metrics["val_loss"])
        if val_metric_for_best < best_val_loss:
            best_val_loss = val_metric_for_best
            best_epoch = epoch
            
            if cfg.checkpoint.save_best:
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model, optimizer, epoch, val_metrics["val_loss"],
                    config=OmegaConf.to_container(cfg, resolve=True),
                    scheduler=scheduler,
                )
        
        if cfg.checkpoint.save_freq and epoch % cfg.checkpoint.save_freq == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch}.pt",
                model, optimizer, epoch, val_metrics["val_loss"],
                scheduler=scheduler,
            )
        
        if early_stopping is not None:
            if early_stopping(val_metric_for_best):
                log.info(f"Early stopping at epoch {epoch}")
                break
    
    if cfg.checkpoint.save_last:
        save_checkpoint(
            ckpt_dir / "last.pt",
            model, optimizer, epoch, val_metrics["val_loss"],
            config=OmegaConf.to_container(cfg, resolve=True),
            scheduler=scheduler,
        )
    
    log.info("=" * 95)
    if is_dual_head:
        log.info(f"Best val ZZ MSE: {best_val_loss:.2e} at epoch {best_epoch}")
    else:
        log.info(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    if test_loader is not None:
        log.info("Running test evaluation...")
        
        if cfg.checkpoint.save_best and (ckpt_dir / "best.pt").exists():
            checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        
        test_metrics = test(model, test_loader, criterion, device, is_graph)
        
        if is_dual_head:
            log.info(f"Test ZZ MSE:   {test_metrics.get('test_zz_mse', 0):.2e}")
            log.info(f"Test Mag Loss: {test_metrics['test_mag_loss']:.2e}")
            log.info(f"Test Sign Acc: {test_metrics['test_sign_acc']:.4f}")
        else:
            log.info(f"Test Loss: {test_metrics['test_loss']:.6f}")
            log.info(f"Test Rel Loss: {test_metrics['test_rel_loss']:.6f}")
        
        if wandb_run:
            import wandb
            wandb.log({
                "test/loss": test_metrics["test_loss"],
                "test/rel_loss": test_metrics["test_rel_loss"],
            })
            if "test_zz_mse" in test_metrics:
                wandb.log({
                    "test/zz_mse": test_metrics["test_zz_mse"],
                    "test/sign_acc": test_metrics["test_sign_acc"],
                    "test/mag_loss": test_metrics["test_mag_loss"],
                })
            wandb.summary["best_val_zz_mse"] = best_val_loss
            wandb.summary["best_epoch"] = best_epoch
    
    if wandb_run:
        import wandb
        wandb.finish()
    
    log.info("Done!")
    return best_val_loss


if __name__ == "__main__":
    main()
