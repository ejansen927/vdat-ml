#!/usr/bin/env python3
"""
Find Best Checkpoint for Toy Case
==================================

Scans outputs/ for all best.pt files, runs inference.py on each using
the real .dat files, and ranks by MAE on r=0 (toy corner case).

Usage:
    python scripts/find_best_checkpoint.py
    python scripts/find_best_checkpoint.py --outputs_dir outputs/ --dat_dir data/
"""

import argparse
import re
import subprocess
from pathlib import Path


def find_checkpoints(outputs_dir: Path):
    """Find all best.pt files."""
    return sorted(outputs_dir.rglob("best.pt"))


def run_inference_and_get_mae(checkpoint: Path, dat_dir: Path, r_values: list, device: str = "cpu"):
    """
    Run inference.py on checkpoint and parse MAE for each r value.
    
    Returns: {r: mae} dict or None if failed
    """
    script_dir = Path(__file__).parent
    
    cmd = [
        "python", str(script_dir / "inference.py"),
        str(checkpoint),
        str(dat_dir),
        "--r"] + [str(r) for r in r_values] + [
        "--device", device,
        "--terminal-only"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
    except Exception as e:
        return None, str(e)
    
    # Parse MAE for each r value
    # Output format: "  Error:   MSE=0.000123  MAE=0.004567"
    results = {}
    current_r = None
    
    for line in output.split('\n'):
        # Detect r value: "Processing r=0..."
        if 'Processing r=' in line:
            match = re.search(r'Processing r=([\d.]+?)\.\.\.', line)
            if match:
                current_r = float(match.group(1))
        
        # Parse error line
        if 'MAE=' in line and current_r is not None:
            mae_match = re.search(r'MAE=([\d.]+)', line)
            if mae_match:
                results[current_r] = float(mae_match.group(1))
    
    return results, None


def main():
    parser = argparse.ArgumentParser(description="Find best checkpoint for toy case")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory with checkpoints")
    parser.add_argument("--dat_dir", default="data", help="Directory with .dat files")
    parser.add_argument("--r", type=float, nargs="+", default=[0, 0.05, 0.25], help="r values")
    parser.add_argument("--sort_by", type=float, default=0, help="r value to sort by")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--top_k", type=int, default=10, help="Show top K")
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    dat_dir = Path(args.dat_dir)
    
    print("=" * 80)
    print("FIND BEST CHECKPOINT")
    print("=" * 80)
    print(f"  Outputs: {outputs_dir}")
    print(f"  Dat dir: {dat_dir}")
    print(f"  r values: {args.r}")
    print(f"  Sort by: r={args.sort_by}")
    print()
    
    # Check dat files exist
    for r in args.r:
        dat_file = dat_dir / f"output_{r}.dat"
        if not dat_file.exists():
            print(f"ERROR: {dat_file} not found")
            return
    
    # Find checkpoints
    checkpoints = find_checkpoints(outputs_dir)
    print(f"Found {len(checkpoints)} checkpoints")
    print()
    
    if not checkpoints:
        print("No best.pt files found!")
        return
    
    # Evaluate each
    results = []
    for i, ckpt in enumerate(checkpoints):
        rel_path = ckpt.relative_to(outputs_dir)
        print(f"[{i+1}/{len(checkpoints)}] {rel_path}...", end=" ", flush=True)
        
        metrics, error = run_inference_and_get_mae(ckpt, dat_dir, args.r, args.device)
        
        if metrics is None or args.sort_by not in metrics:
            print(f"FAILED ({error})")
            continue
        
        mae = metrics[args.sort_by]
        print(f"MAE(r={args.sort_by})={mae:.6f}")
        results.append((mae, metrics, ckpt, rel_path))
    
    if not results:
        print("\nNo valid results!")
        return
    
    # Sort by MAE
    results.sort(key=lambda x: x[0])
    
    # Print table
    print("\n" + "=" * 80)
    print(f"TOP {min(args.top_k, len(results))} CHECKPOINTS (sorted by MAE at r={args.sort_by})")
    print("=" * 80)
    
    header = "Rank  "
    for r in args.r:
        header += f"r={r:<8}"
    header += "  Checkpoint"
    print(header)
    print("-" * 80)
    
    for rank, (_, metrics, full_path, rel_path) in enumerate(results[:args.top_k], 1):
        row = f"{rank:<6}"
        for r in args.r:
            if r in metrics:
                mae = metrics[r]
                if r == args.sort_by:
                    row += f"\033[1m{mae:<10.6f}\033[0m"
                else:
                    row += f"{mae:<10.6f}"
            else:
                row += "N/A       "
        row += f"  {rel_path}"
        print(row)
    
    # Best checkpoint
    best_mae, best_metrics, best_path, best_rel = results[0]
    
    print("\n" + "=" * 80)
    print("BEST CHECKPOINT")
    print("=" * 80)
    print(f"  Path: {best_path}")
    print()
    for r in args.r:
        if r in best_metrics:
            marker = " ← TOY CORNER (J=1)" if r == 0 else ""
            print(f"  r={r}: MAE={best_metrics[r]:.6f}{marker}")
    
    print("\n" + "=" * 80)
    print("Run inference:")
    print(f"  python scripts/inference.py {best_path} {dat_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
