for expected_batchsize in 2048
do
for epoch in 5
do
for lr in 0.001
do
for worker_num in 16
do
for C in 1
do
for graph_setting in naive
do
for priv_epsilon in {2,4,8,16}
do
for K in 1
do
for num_neighbors in 1
do
for num_neighbors_test in 1
do
for num_not_neighbors in 1
do
for seed in {1,2,3,4,5}
do
for dataset in {facebook,twitch_DE,Reddit,Amazon_Computers,PubMed}
do

python main_NaiveDPSGD.py --expected_batchsize $expected_batchsize \
                --priv_epsilon $priv_epsilon \
                --epoch $epoch \
                --lr $lr \
                --log_dir logs \
                --K $K \
                --num_neighbors $num_neighbors \
                --num_neighbors_test $num_neighbors_test \
                --num_not_neighbors $num_not_neighbors \
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


