#! /bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 feature_corrs.py