# VDAT ML - Handbook

Quick reference for running experiments.

---

## Data Generation

```bash
# Basic: 1000 samples, 4 qubits
python scripts/generate_data.py --dist random --n_samples 1000 --n_qubits 4

# Large dataset with auto-timestamp (safe, won't overwrite)
python scripts/generate_data.py --dist random --n_samples 10000 --n_qubits 8 --append_timestamp

# Custom output location
python scripts/generate_data.py --dist random --n_samples 5000 --n_qubits 4 --output data/experiment_v2
```

Output:
```
data/random_4q/
├── train.pt      (800 samples)
├── val.pt        (100 samples)
├── test.pt       (100 samples)
└── metadata.yaml

configs/data/random_4q.yaml        # MLP config
configs/data/random_4q_graph.yaml  # GNN config
```

---

## Training

### Quick Commands

```bash
# Debug (5 epochs, no W&B, fast)
python src/train.py experiment=debug

# MLP on your data
python src/train.py data=random_4q model=mlp

# GNN on your data
python src/train.py data=random_4q_graph model=gnn

# Production run (500 epochs, full logging)
python src/train.py experiment=production data=random_4q model=mlp
```

### Override Anything

```bash
# Learning rate
python src/train.py optimizer.lr=1e-4

# Architecture
python src/train.py model=mlp model.hidden_dims=[64,128,256,128,64]
python src/train.py model=gnn model.hidden_dim=256 model.num_layers=5

# Training
python src/train.py training.epochs=200 training.batch_size=64

# Combine
python src/train.py model=gnn optimizer=adamw optimizer.lr=5e-4 training.epochs=300
```

### Sweeps (Multirun)

```bash
# Compare models
python src/train.py -m model=mlp,gnn data=random_4q_graph

# Learning rate sweep
python src/train.py -m optimizer.lr=1e-3,5e-4,1e-4

# Architecture search
python src/train.py -m model.hidden_dim=64,128,256 model.num_layers=2,3,4

# Full grid (runs all combinations)
python src/train.py -m model=mlp,gnn optimizer.lr=1e-3,1e-4 training.batch_size=32,64
```

---

## Example Experiments

### 1. Baseline Comparison

Compare MLP vs GNN on same data:

```bash
# Generate data
python scripts/generate_data.py --dist random --n_samples 5000 --n_qubits 4

# Run both
python src/train.py data=random_4q model=mlp project.experiment=baseline
python src/train.py data=random_4q_graph model=gnn project.experiment=baseline
```

Check W&B dashboard → filter by `experiment=baseline`.

### 2. Scaling Study

How does performance change with more qubits?

```bash
# Generate datasets
python scripts/generate_data.py --dist random --n_samples 5000 --n_qubits 4
python scripts/generate_data.py --dist random --n_samples 5000 --n_qubits 6
python scripts/generate_data.py --dist random --n_samples 5000 --n_qubits 8

# Train GNN on each
python src/train.py data=random_4q_graph model=gnn project.experiment=scaling
python src/train.py data=random_6q_graph model=gnn project.experiment=scaling
python src/train.py data=random_8q_graph model=gnn project.experiment=scaling
```

### 3. Hyperparameter Search

Find optimal learning rate:

```bash
python src/train.py -m \
    data=random_4q_graph \
    model=gnn \
    optimizer.lr=1e-2,5e-3,1e-3,5e-4,1e-4 \
    project.experiment=lr_search
```

### 4. Architecture Ablation

Test GNN depth and width:

```bash
python src/train.py -m \
    model=gnn \
    model.hidden_dim=64,128,256 \
    model.num_layers=2,3,4,5 \
    project.experiment=architecture
```

### 5. Quick Sanity Check

Before a long run, verify everything works:

```bash
python src/train.py experiment=debug data=random_4q model=mlp
```

---

## Evaluation

```bash
# Evaluate saved checkpoint
python src/evaluate.py checkpoint=outputs/.../checkpoints/best.pt data=random_4q model=mlp
```

---

## W&B Tips

- **Compare runs:** Use the table view, add columns for `model.name`, `optimizer.lr`, `test_loss`
- **Filter:** `experiment=baseline` shows only runs from that experiment
- **Group:** Group by `model.name` to see MLP vs GNN side by side

---

## Config Reference

| Config | Options |
|--------|---------|
| `model` | `mlp`, `gnn` |
| `optimizer` | `adam`, `adamw`, `sgd` |
| `scheduler` | `cosine`, `plateau`, `none` |
| `experiment` | `debug`, `quick`, `production` |

Key parameters:
```yaml
training.epochs: 500
training.batch_size: 32
training.loss: mse  # or mae
optimizer.lr: 1e-3
model.hidden_dims: [128, 256, 128]  # MLP
model.hidden_dim: 128               # GNN
model.num_layers: 3                 # GNN
model.activation: silu              # silu, gelu, relu
model.dropout: 0.0
```

---

## Troubleshooting

**CUDA out of memory:**
```bash
python src/train.py training.batch_size=16
```

**W&B disabled:**
```bash
python src/train.py logging.wandb.enabled=false
```

**See full config:**
```bash
python src/train.py --cfg job
```
