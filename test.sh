#!/bin/bash
# my sample tests

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
