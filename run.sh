#!/bin/bash
#

python src/train.py \
    model=gnn \
    data=Niraj1M_graph \
    model.num_layers=4 \
    model.hidden_dim=256 \
    optimizer=adamw \
    optimizer.lr=1.5e-3 \
    optimizer.weight_decay=1e-3 \
    training.epochs=100 \
    training.batch_size=256 \
    model.pooling=sum \
    project.name=gnn-testing \
    project.experiment=gnn_al1
