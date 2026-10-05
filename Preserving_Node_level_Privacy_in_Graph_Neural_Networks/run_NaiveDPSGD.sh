#!/usr/bin/env bash
# Baseline: DP-SGD on an MLP over node features only (no graph structure).
for priv_epsilon in 2 4 8 16; do
for seed in 1 2 3 4 5; do
for dataset in facebook twitch_DE Reddit Amazon_Computers PubMed; do
    python main_NaiveDPSGD.py --dataset $dataset \
        --expected_batchsize 2048 --epoch 5 --lr 0.001 --C 1 \
        --worker_num 16 --log_dir logs \
        --priv_epsilon $priv_epsilon \
        --seed $seed
done
done
done
