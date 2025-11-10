# SFL with cyclic order with Dominand Label partition

python run_sfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method dl \
                --num_shards_per_partition 1 --policy cyclic --cut 4  --alpha 4\
                --batch_size 64 --device mps --phi 10

# SFL with cyclic order with Dominand Label partition with superclasses L=20

python run_sfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar100  --epoch 100 \
                --training_lr 0.05 --num_client 20 --partition_method dl \
                --num_shards_per_partition 1 --policy cyclic --cut 4  --alpha 4\
                --batch_size 64 --device mps --phi 10 --classif_target coarse_label

# SFL with cyclic order with Dominand Label partition with all classes L=100

python run_sfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar100  --epoch 100 \
                --training_lr 0.05 --num_client 100 --partition_method dl \
                --num_shards_per_partition 1 --policy cyclic --cut 4  --alpha 4 \
                --batch_size 64 --device mps --phi 10 --classif_target fine_label

# SFL with cyclic order with sharding partition 

python run_sfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method sharding \
                --num_shards_per_partition 1 --policy random --cut 4  --alpha 4\
                --batch_size 64 --device mps --phi 10

# SFL with cyclic order with dirchlet partition

python run_sfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method dirichlet \
                --num_shards_per_partition 1 --policy random --cut 4  --alpha 0.1 \
                --batch_size 64 --device mps --phi 10


# SFL with reverse order with DL partition

python run_sfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method dl \
                --num_shards_per_partition 1 --policy reverse --cut 4  --alpha 4\
                --batch_size 64 --device mps --phi 10


# SFL with random order with DL partition

python run_sfl.py --model_name mobilenet  --model_name mobilenet --dataset_type cifar10  --epoch 100 \
                --training_lr 0.05 --num_client 10 --partition_method dl \
                --num_shards_per_partition 1 --policy random --cut 4  --alpha 4\
                --batch_size 64 --device mps --phi 10