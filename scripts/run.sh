#!/bin/bash
echo """
# 1-3: GNN Learning Rate (3 runs)
python src/train.py -m \
    model=gnn \
    data=random_4q_graph \
    optimizer=adamw \
    optimizer.lr=1e-3,1.5e-3,2e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    project.name=VDAT-ML \
    project.experiment=gnn_lr

# 4-6: MLP Learning Rate (3 runs)
python src/train.py -m \
    model=mlp \
    data=random_4q \
    optimizer=adamw \
    optimizer.lr=1e-3,1.5e-3,2e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    project.name=VDAT-ML \
    project.experiment=mlp_lr

# 7-9: GNN Depth (3 runs)
python src/train.py -m \
    model=gnn \
    data=random_4q_graph \
    model.num_layers=2,3,4 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    project.name=VDAT-ML \
    project.experiment=gnn_depth

# 10-12: GNN Hidden Dim (3 runs)
python src/train.py -m \
    model=gnn \
    data=random_4q_graph \
    model.hidden_dim=128,256,384 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    project.name=VDAT-ML \
    project.experiment=gnn_width

# 13-15: MLP Architecture (3 runs)
python src/train.py -m \
    model=mlp \
    data=random_4q \
    "model.hidden_dims=[128,256,128],[256,512,256],[256,256,256]" \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    project.name=VDAT-ML \
    project.experiment=mlp_arch"""

echo "fair parameters"
"""
# GNN baseline: 2 configs
python src/train.py -m \
    model=gnn \
    data=random_4q_graph \
    model.num_layers=4 \
    model.hidden_dim=128 \
    optimizer=adamw \
    optimizer.lr=1e-3,1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    project.name=VDAT-ML \
    project.experiment=fair_gnn

# MLP matched params (~300K):
python src/train.py -m \
    model=mlp \
    data=random_4q \
    "model.hidden_dims=[384,384,384],[320,320,320,320]" \
    optimizer=adamw \
    optimizer.lr=1e-3,1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    project.name=VDAT-ML \
    project.experiment=fair_mlp"""
"""
# GNN with new seed
python src/train.py -m \
    model=gnn \
    data=random_4q_graph \
    model.num_layers=4 \
    model.hidden_dim=128 \
    optimizer=adamw \
    optimizer.lr=1e-3,1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    training.seed=123 \
    project.name=VDAT-ML \
    project.experiment=fair_gnn_seed2

# MLP with new seed
python src/train.py -m \
    model=mlp \
    data=random_4q \
#    "model.hidden_dims=[384,384,384],[320,320,320,320]" \
    optimizer=adamw \
    optimizer.lr=1e-3,1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64 \
    training.seed=123 \
    project.name=VDAT-ML \
    project.experiment=fair_mlp_seed2"""
