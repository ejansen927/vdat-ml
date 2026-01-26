#!/bin/bash

# MLP: batch size × hidden dims sweep (9 runs)
python src/train.py -m \
    model=mlp \
    data=sobol_4q \
    "model.hidden_dims=[256,256,256],[384,384,384],[320,320,320,320]" \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64,128,256 \
    project.name=VDAT-ML \
    project.experiment=mlp_sobol1

# GNN: batch size × hidden dim sweep (9 runs)
python src/train.py -m \
    model=gnn \
    data=sobol_4q_graph \
    model.num_layers=4 \
    model.hidden_dim=64,128,256 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64,128,256 \
    project.name=VDAT-ML \
    project.experiment=gnn_sobol1
