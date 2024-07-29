#!/bin/bash

# configuration
dataset='gaia'
epochs=3000
lr=0.001
batch_size=128
guide_weight=0.1
aug_percent=0.2

if [ "$dataset" = "gaia" ]; then
    python main.py --dataset "gaia" --N_I 10 --N_T 5 --temperature 0.3  --epochs $epochs --lr $lr --batch_size $batch_size --aggregator "lstm" --guide_weight $guide_weight --patience 10 --aug_percent $aug_percent --dynamic_weight --TO --CM
fi