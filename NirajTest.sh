#!/bin/bash

data="combined_6.1M_graph"

python src/train.py \
    model=gnn \
    model.name=dual_head_gnn \
    data=$data \
    model.num_layers=4 \
    model.hidden_dim=128 \
    +model.n_qubits_pe=4 \
    optimizer.lr=5e-4 \
    optimizer.weight_decay=1e-4 \
    training.epochs=100 \
    training.batch_size=4096 \
    scheduler=cosine \
    scheduler.T_max=100 \
    model.pooling=mean \
    project.experiment=random_gnn_test \
    project.name=gnn-testing \
    training.early_stopping.enabled=True \
    training.early_stopping.patience=20
