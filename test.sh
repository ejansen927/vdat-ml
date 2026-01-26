#!/bin/bash
#


# GNN: batch size × hidden dim sweep (9 runs, ~9 hrs)
python src/train.py -m \
    model=gnn \
    data=random_4q_graph \
    model.num_layers=4 \
    model.hidden_dim=64,128,256 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64,128,256 \
    project.name=VDAT-ML \
    project.experiment=gnn_batch_width
