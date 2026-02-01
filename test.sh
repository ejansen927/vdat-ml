#!/bin/bash
#
"""
# Run 1: AL uncertainty + random candidates
python src/train.py \
    model=gnn \
    data=al_uncertainty1 \
    model.num_layers=4 \
    model.hidden_dim=256 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=256 \
    project.name=VDAT-ML \
    project.experiment=gnn_al_uncertainty1

# Run 2: AL uncertainty + clipped candidates
python src/train.py \
    model=gnn \
    data=al_uncertainty2 \
    model.num_layers=4 \
    model.hidden_dim=256 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=256 \
    project.name=VDAT-ML \
    project.experiment=gnn_al_uncertainty2

python src/train.py \
    model=gnn \
    data=hybrid_sweep_100k_graph \
    model.num_layers=4 \
    model.hidden_dim=256 \
    model.dropout=0.1 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=256 \
    project.name=VDAT-ML \
    project.experiment=gnn_hybrid_sweep"""

python src/train.py -m \
    model=gnn \
    data=sobol_4q_graph \
    model.num_layers=4 \
    model.hidden_dim=256 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=256 \
    training.loss=mse,mae,huber \
    project.name=VDAT-ML \
    project.experiment='gnn_loss_${training.loss}'
