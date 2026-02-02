# VDAT ML - Handbook

Quick reference for running experiments.

---

## Exact Diagonalization Data Generation with Transverse Field Ising Model

```bash
# for a basic test: 1000 samples, 4 qubits
python scripts/generate_data.py --dist random --n_samples 1000 --n_qubits 4

# here can append timestamp, wont overwrite existing data
python scripts/generate_data.py --dist random --n_samples 10000 --n_qubits 8 --append_timestamp

# add this simply to overwrite data
--overwrite

# customize data name
python scripts/generate_data.py --dist random --n_samples 5000 --n_qubits 4 --output data/experiment_v2

# 

```
Available data distributions:
- random
- sobol (quasi-random, fills space evenly)
- normal

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
# Run a quick debugging experiment (5 epochs)
python src/train.py experiment=debug

# Test the MLP (basic feed forward NN) model
python src/train.py data=random_4q model=mlp

# Run the GNN model
python src/train.py data=random_4q_graph model=gnn

# Production run set to 500 epochs for now
python src/train.py experiment=production data=random_4q model=mlp
```

### Other settings

```bash
# set learning rate
python src/train.py optimizer.lr=1e-4

# style the mlp/gnn architecture
python src/train.py model=mlp model.hidden_dims=[64,128,256,128,64]
python src/train.py model=gnn model.hidden_dim=256 model.num_layers=5

# define training epochs/batch size
python src/train.py training.epochs=200 training.batch_size=64

# optimizer too
python src/train.py model=gnn optimizer=adamw optimizer.lr=5e-4 training.epochs=300
```

### Sweeps

Can run sweep jobs as a grid-search with '-m' and defining multiple values for selected keywords, ie, model=gnn,mlp sweeps over both models.

```bash
# Compare models
python src/train.py -m model=mlp,gnn data=random_4q_graph

# lr example
python src/train.py -m optimizer.lr=1e-3,5e-4,1e-4

# arch.
python src/train.py -m model.hidden_dim=64,128,256 model.num_layers=2,3,4

# larger grid search
python src/train.py -m model=mlp,gnn optimizer.lr=1e-3,1e-4 training.batch_size=32,64
```

---

## Example experiment

Compare MLP vs GNN on same data, and set the project name (this will appear in my W&B, and I can add team members to it). Just make a bash script and run jobs very simply like this in a tmux session.

```bash
# make a dataset
python scripts/generate_data.py --dist random --n_samples 5000 --n_qubits 4

# run projects
python src/train.py data=random_4q model=mlp project.experiment=baseline
python src/train.py data=random_4q_graph model=gnn project.experiment=baseline
```

Check W&B dashboard and filter for the project.experiment=baseline in "Runs".

## Evaluation

Runs a saved model (outputs/project_name/checkpoint/best.pt) on a test set.

```bash
# Evaluate saved checkpoint
python src/evaluate.py checkpoint=outputs/.../checkpoints/best.pt data=random_4q model=mlp
```

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
training.loss: mse  # takes mse, mae, huber
optimizer.lr: 1e-3
model.hidden_dims: [128, 256, 128]  # only for mlp
model.hidden_dim: 128               # only for gnn (hidden dim used for mlps inside)
model.num_layers: 3                 # only gnn number of layers
model.activation: silu              # silu, gelu, relu
model.dropout: 0.0               
```
