#!/usr/bin/env python3
"""
Inspect a checkpoint file and display all saved information.

Usage:
    python scripts/inspect_checkpoint.py checkpoints/best.pt
    python scripts/inspect_checkpoint.py outputs/gnn_lr/multirun/.../checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch


def format_value(v, indent=0):
    """Format a value for display."""
    prefix = "  " * indent
    
    if isinstance(v, dict):
        lines = []
        for k, val in v.items():
            formatted = format_value(val, indent + 1)
            if "\n" in formatted:
                lines.append(f"{prefix}{k}:")
                lines.append(formatted)
            else:
                lines.append(f"{prefix}{k}: {formatted}")
        return "\n".join(lines)
    elif isinstance(v, (list, tuple)):
        if len(v) <= 5 and all(not isinstance(x, (dict, list, tuple)) for x in v):
            return str(list(v))
        else:
            lines = []
            for i, item in enumerate(v):
                lines.append(f"{prefix}[{i}]: {format_value(item, indent + 1)}")
            return "\n".join(lines)
    elif isinstance(v, torch.Tensor):
        return f"Tensor{list(v.shape)} dtype={v.dtype}"
    else:
        return str(v)


def inspect_checkpoint(path: str):
    """Load and display checkpoint contents."""
    
    # Colors
    BOLD = '\033[1m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Checkpoint: {path}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")
    
    # Load checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Basic info
    print(f"{BLUE}{BOLD}[Keys in checkpoint]{RESET}")
    for key in ckpt.keys():
        print(f"  • {key}")
    print()
    
    # Epoch and loss
    if "epoch" in ckpt:
        print(f"{GREEN}{BOLD}[Training Info]{RESET}")
        print(f"  Epoch:    {ckpt['epoch']}")
        if "val_loss" in ckpt:
            print(f"  Val Loss: {ckpt['val_loss']:.6f}")
        print()
    
    # Config (new format)
    if "config" in ckpt or "cfg" in ckpt:
        cfg = ckpt.get("config") or ckpt.get("cfg")
        print(f"{GREEN}{BOLD}[Configuration]{RESET}")
        
        # Model config
        if "model" in cfg:
            print(f"\n  {YELLOW}Model:{RESET}")
            model_cfg = cfg["model"]
            for k, v in model_cfg.items():
                if not k.startswith("_"):
                    print(f"    {k}: {v}")
        
        # Data config
        if "data" in cfg:
            print(f"\n  {YELLOW}Data:{RESET}")
            data_cfg = cfg["data"]
            for k, v in data_cfg.items():
                if not k.startswith("_"):
                    print(f"    {k}: {v}")
        
        # Optimizer config
        if "optimizer" in cfg:
            print(f"\n  {YELLOW}Optimizer:{RESET}")
            opt_cfg = cfg["optimizer"]
            for k, v in opt_cfg.items():
                if not k.startswith("_"):
                    print(f"    {k}: {v}")
        
        # Scheduler config
        if "scheduler" in cfg:
            print(f"\n  {YELLOW}Scheduler:{RESET}")
            sched_cfg = cfg["scheduler"]
            for k, v in sched_cfg.items():
                if not k.startswith("_"):
                    print(f"    {k}: {v}")
        
        # Training config
        if "training" in cfg:
            print(f"\n  {YELLOW}Training:{RESET}")
            train_cfg = cfg["training"]
            for k, v in train_cfg.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        print(f"      {k2}: {v2}")
                else:
                    print(f"    {k}: {v}")
        
        print()
    
    # Model kwargs (old format)
    if "model_kwargs" in ckpt:
        print(f"{GREEN}{BOLD}[Model Kwargs (old format)]{RESET}")
        for k, v in ckpt["model_kwargs"].items():
            print(f"  {k}: {v}")
        print()
    
    # Model state dict summary
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    if state_key in ckpt:
        state = ckpt[state_key]
        print(f"{GREEN}{BOLD}[Model Architecture]{RESET}")
        
        # Count parameters
        total_params = 0
        layer_info = {}
        
        for name, param in state.items():
            total_params += param.numel()
            
            # Group by layer
            parts = name.split(".")
            if len(parts) >= 2:
                layer_name = ".".join(parts[:2])
            else:
                layer_name = parts[0]
            
            if layer_name not in layer_info:
                layer_info[layer_name] = {"params": 0, "shapes": []}
            layer_info[layer_name]["params"] += param.numel()
            layer_info[layer_name]["shapes"].append((name, list(param.shape)))
        
        print(f"\n  {YELLOW}Total Parameters:{RESET} {total_params:,}")
        print(f"\n  {YELLOW}Layers:{RESET}")
        
        for layer_name, info in layer_info.items():
            print(f"    {layer_name}: {info['params']:,} params")
            for name, shape in info["shapes"][:3]:  # Show first 3 tensors
                print(f"      {GRAY}{name}: {shape}{RESET}")
            if len(info["shapes"]) > 3:
                print(f"      {GRAY}... and {len(info['shapes']) - 3} more{RESET}")
        
        print()
    
    # Optimizer state summary
    if "optimizer_state_dict" in ckpt:
        opt_state = ckpt["optimizer_state_dict"]
        print(f"{GREEN}{BOLD}[Optimizer State]{RESET}")
        print(f"  Param groups: {len(opt_state.get('param_groups', []))}")
        if opt_state.get("param_groups"):
            pg = opt_state["param_groups"][0]
            print(f"  Learning rate: {pg.get('lr', 'N/A')}")
            print(f"  Weight decay: {pg.get('weight_decay', 'N/A')}")
            print(f"  Betas: {pg.get('betas', 'N/A')}")
        print()
    
    # Scheduler state
    if "scheduler_state_dict" in ckpt:
        sched_state = ckpt["scheduler_state_dict"]
        print(f"{GREEN}{BOLD}[Scheduler State]{RESET}")
        for k, v in sched_state.items():
            if not isinstance(v, (list, dict)) or len(str(v)) < 50:
                print(f"  {k}: {v}")
        print()
    
    # File size
    file_size = Path(path).stat().st_size
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024*1024):.2f} MB"
    elif file_size > 1024:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size} bytes"
    
    print(f"{GRAY}File size: {size_str}{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint (.pt)")
    parser.add_argument("--raw", action="store_true", help="Show raw dict structure")
    args = parser.parse_args()
    
    path = Path(args.checkpoint)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)
    
    if args.raw:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        print(format_value(ckpt))
    else:
        inspect_checkpoint(str(path))


if __name__ == "__main__":
    main()
