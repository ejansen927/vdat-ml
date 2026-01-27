#!/bin/bash

python scripts/generate_data.py \
    --dist hybrid \
    --n_samples 1000000 \
    --n_qubits 4 \
    --sobol_fraction 0.5 \
    --boundary_width 0.1 \
    --chunk_size 25000
