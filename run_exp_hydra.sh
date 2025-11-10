# Hydra with cyclic order with Dominand Label partition 

python run_hydra.py --model_name mobilenet --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method dl \
                --num_shards_per_partition 1 --policy cyclic --cut_a 4  --cut_b 23 --alpha 4\
                --batch_size 64 --device mps --phi 10


# Hydra with cyclic order with Dominand Label partition CIFAR-10 changing the number of heads

python test_hydra_heads.py --model_name mobilenet --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method dl \
                --num_shards_per_partition 1 --policy cyclic --cut_a 4  --cut_b 23 --alpha 4\
                --batch_size 64 --device mps --phi 10 --num_heads 5