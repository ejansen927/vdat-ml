import torch
from pathlib import Path

chunk_dir = Path("data/combined_6.1M/chunks")
out_dir = Path("data/combined_6.1M")

for split in ["train", "val", "test"]:
    chunks = sorted(chunk_dir.glob(f"{split}_*.pt"))
    if not chunks:
        continue
    
    parts = [torch.load(c, weights_only=True) for c in chunks]
    merged = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    torch.save(merged, out_dir / f"{split}.pt")
