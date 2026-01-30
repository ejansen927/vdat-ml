#!/bin/bash
#

python src/train.py -m \
    model=gnn \
    data=hybrid_4q_graph \
    model.num_layers=4 \
    model.hidden_dim=64,128,256 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=64,128,256 \
    project.name=VDAT-ML \
    project.experiment=gnn_hybrid1
