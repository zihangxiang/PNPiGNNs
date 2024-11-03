for expected_batchsize in 4096
do
for epoch in 4
do
for lr in 0.01
do
for worker_num in 16
do
for C in 1
do
for graph_setting in {transductive,inductive}
do
for priv_epsilon in 2
do
for K in 1
do
for num_neighbors in {1,2,3,4,5}
do
for num_neighbors_test in {1,4,7,10,13}
do
for num_not_neighbors in 0
do
for seed in {1,2,3}
do
for dataset in Reddit
do

python main.py --expected_batchsize $expected_batchsize \
                --priv_epsilon $priv_epsilon \
                --epoch $epoch \
                --lr $lr \
                --log_dir logs \
                --K $K \
                --num_neighbors $num_neighbors \
                --num_not_neighbors $num_not_neighbors \
                --num_neighbors_test $num_neighbors_test \
                --worker_num $worker_num \
                --C $C \
                --seed $seed \
                --graph_setting $graph_setting \
                --dataset $dataset 
done
done
done
done
done
done
done
done
done
done
done
done
done


