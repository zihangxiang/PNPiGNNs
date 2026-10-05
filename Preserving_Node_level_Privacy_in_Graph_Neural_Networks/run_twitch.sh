#!/usr/bin/env bash
# Node-level DP GNN on twitch_DE: hyper-parameter sweep.
# priv_epsilon=10000 serves as a (practically) non-private reference.
for graph_setting in transductive inductive; do
for priv_epsilon in 2 4 8 16 10000; do
for num_neighbors in 1 2 3 4 5; do
for num_neighbors_test in 1 4 7 10 13; do
for seed in 1 2 3; do
    python main.py --dataset twitch_DE \
        --expected_batchsize 4096 --epoch 9 --lr 0.01 --C 1 --K 1 \
        --worker_num 16 --log_dir logs \
        --graph_setting $graph_setting \
        --priv_epsilon $priv_epsilon \
        --num_neighbors $num_neighbors \
        --num_neighbors_test $num_neighbors_test \
        --seed $seed
done
done
done
done
done
