# SplitFedv1/FL

# python run_fl.py --model_name mobilenet --model_name mobilenet --dataset_type cifar10  --epoch 100 \
#                 --training_lr 0.05 --num_client 10 --partition_method dl \
#                 --num_shards_per_partition 1 --alpha 4\
#                 --batch_size 64 --device mps

# SplitFedv3

# python run_sflv3.py --model_name mobilenet --model_name mobilenet --dataset_type cifar10  --epoch 100 \
#                 --training_lr 0.05 --num_client 10 --partition_method dl --cut 4 \
#                 --num_shards_per_partition 1 --alpha 4\
#                 --batch_size 64 --device mps


# MultiHead

python run_multihead.py --model_name mobilenet --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method dl --cut 23 \
                --num_shards_per_partition 1 --alpha 4\
                --batch_size 64 --device mps

# MergeSFL

# python run_mergesfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar10  --epoch 100 \
#                 --training_lr 0.05 --num_client 10 --partition_method dl \
#                 --num_shards_per_partition 1 --cut 4  --alpha 4\
#                 --batch_size 64 --device mps