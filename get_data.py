import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, ShardPartitioner, DistributionPartitioner

############# USING FLOWER API START #############

def generate_partitioner(partion_args):
    ## IID
    if partion_args['partition_method'] == 'iid':
        method = 'iid',
        partition_by=partion_args['partition_target'],    
        partitioner = IidPartitioner(num_partitions=partion_args['num_clients'])           
        
    ## DIRICHLET    
    elif partion_args['partition_method'] == 'dirichlet':
        method = 'dirichlet'
        partitioner = DirichletPartitioner(
            num_partitions=partion_args['num_clients'],
            partition_by=partion_args['partition_target'],
            alpha=partion_args['alpha'],
            seed=partion_args.get('dataset_seed', 42),
            min_partition_size=0)

    elif partion_args['partition_method'] in ['distribution', 'special']:
        method = 'distribution'
        if partion_args['alpha'] == 4:
            if 'cifar100' in partion_args['dataset_name'] and partion_args['partition_target'] == 'coarse_label':
                distribution_array = np.zeros((20, 20))
                num_unique_labels_per_partition = 20
                preassigned_num_samples_per_label = 20
                for i in range(20):
                    for j in range(20):
                        if i==j:
                            distribution_array[i][j] = 0.5
                        else:
                            distribution_array[i][j] = 0.03
            elif 'cifar100' in partion_args['dataset_name'] and partion_args['partition_target'] == 'fine_label':
                distribution_array = np.zeros((100, 100))
                num_unique_labels_per_partition = 10
                preassigned_num_samples_per_label = 10
                for i in range(100):
                    for j in range(100):
                        if i==j:
                            distribution_array[i][j] = 0.5
                        else:
                            distribution_array[i][j] = 0.01
        elif partion_args['alpha'] == 3:
            distribution_array = np.zeros((20, 20))
            num_unique_labels_per_partition = 20
            preassigned_num_samples_per_label = 20
            for i in range(20):
                for j in range(20):
                    if i==j:
                        distribution_array[i][j] = 0.6
                    else:
                        distribution_array[i][j] = 0.021

        partitioner = DistributionPartitioner(
            distribution_array=distribution_array,
            num_partitions=partion_args['num_clients'],
            num_unique_labels_per_partition=num_unique_labels_per_partition,
            partition_by=partion_args['partition_target'],  
            preassigned_num_samples_per_label=preassigned_num_samples_per_label,
        )
        
    ## ARCHETYPE
    else:
        method = 'archetype'
        partitioner = ShardPartitioner(
            num_partitions=partion_args['num_clients'],
            partition_by=partion_args['partition_target'],
            num_shards_per_partition=partion_args['num_shards_per_partition'],
            #shard_size = partion_args['shard_size']
        )
    return partitioner


def execute_partition_and_plot(partion_args, dataset_type, batch_size=64):
    fds = FederatedDataset(
        dataset=partion_args['dataset_name'],
        partitioners={
            "train": generate_partitioner(partion_args)
        },
    )

    fig, axes, df_list = plot_label_distributions(
        partitioner=fds.partitioners["train"],
        label_name=partion_args['partition_target'],
        title=f"{partion_args['dataset_name']} - {partion_args['partition_method']} - {partion_args['partition_target']}",
        legend=True,
        verbose_labels=True,
    )

    plt.savefig('distribution.png')
    print(df_list)

    def apply_transforms_train_cifar10(batch):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        key_ = 'img'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch
    
    def apply_transforms_train_cifar100(batch):        
        transform =  transforms.Compose(
                        [
                        transforms.RandomCrop(32, 4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                        ])

        key_ = 'img'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch
    
    def apply_transforms_train_tinyimage(batch):
        
        transform =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # Implementing CutMix
            # transforms.Lambda(lambda x: mixup(x) if random.random() > 0.5 else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        key_ = 'image'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch

    
    def apply_transforms_test_cifar10(batch):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize
            ]
        )

        key_ = 'img'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch

    def apply_transforms_test_cifar100(batch):
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize
            ]
        )

        key_ = 'img'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch
    
    def apply_transforms_test_tinyimage(batch):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        key_ = 'image'
        batch[key_] = [transform(img) for img in batch[key_]]
        return batch
    

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(partion_args['num_clients']):
        partition = fds.load_partition(partition_id, "train")

        if dataset_type in ['cifar10', 'tinyImageNet']:
            partition = partition.train_test_split(train_size=0.8, seed=42)
        elif dataset_type == 'cifar100':
             partition = partition.train_test_split(train_size=0.9, seed=42)
        
        if dataset_type == 'cifar10':
            partition["train"] = partition["train"].with_transform(apply_transforms_train_cifar10)
            partition["test"] = partition["test"].with_transform(apply_transforms_test_cifar10)
        elif dataset_type == 'cifar100':
            partition["train"] = partition["train"].with_transform(apply_transforms_train_cifar100)
            partition["test"] = partition["test"].with_transform(apply_transforms_test_cifar100)
        else:
            partition["train"] = partition["train"].with_transform(apply_transforms_train_tinyimage)
            partition["test"] = partition["test"].with_transform(apply_transforms_test_tinyimage)
        
        

        trainloaders.append(DataLoader(partition["train"], batch_size=batch_size))
        valloaders.append(DataLoader(partition["test"], batch_size=batch_size))
    if dataset_type in ['cifar10']:
        testset = fds.load_split("test").with_transform(apply_transforms_test_cifar10)
    if dataset_type == 'cifar100':
        testset = fds.load_split("test").with_transform(apply_transforms_test_cifar100)
    if dataset_type == 'tinyImageNet':
        testset = fds.load_split("validation").with_transform(apply_transforms_test_tinyimage)

    
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader, df_list
############# USING FLOWER API END #############

############# USING MERGESFL PART START #############
## REFERENCE CODE: https://github.com/ymliao98/MergeSFL/blob/main/datasets.py
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataLoaderHelper(object):
    def __init__(self, dataloader):
        self.loader = dataloader
        self.dataiter = iter(self.loader)

    def __next__(self):
        try:
            data, target = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            data, target = next(self.dataiter)
        
        return data, target
    
    def __len__(self):
        return (len(self.loader))
    
    def __iter__(self):
        return iter(self.loader)
    

def load_default_transform(dataset_type, train=False):
    if dataset_type == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        if train:
            dataset_transform = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])

    elif dataset_type == 'cifar100':
        # reference:https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            dataset_transform = transforms.Compose([
                                transforms.RandomCrop(32, 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                normalize
                            ])
        else:
            dataset_transform = transforms.Compose([
                            transforms.ToTensor(),
                            normalize
                            ])
    elif dataset_type == 'SVHN':
        dataset_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    elif dataset_type == 'tinyImageNet':
        dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    else:
        dataset_transform=None

    return dataset_transform

    
class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, class_num=10, labels=None, seed=2020):
        # sizes is a class_num * vm_num matrix
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()
        # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
        if hasattr(data, 'classes'):
            # focus here
            for class_idx in range(len(data.classes)):
                label_indexes.append(list(np.where(np.array(data.targets) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        elif hasattr(data, 'labels'):
            for class_idx in range(class_num): 
                label_indexes.append(list(np.where(np.array(data.labels) == class_idx)[0]))
                class_len.append(len(label_indexes[class_idx]))
                rng.shuffle(label_indexes[class_idx])
        else:
            label_indexes = [list() for _ in range(class_num)]
            class_len = [0] * class_num
            # label_indexes includes class_num lists. Each list is the set of indexs of a specific class
            for i, j in enumerate(data):
                if labels:
                    class_idx = labels.index(j[2])
                else:
                    class_idx = int(j[1])
                class_len[class_idx] += 1
                label_indexes[class_idx].append(i)
            # print(class_size)
            for i in range(class_num):
                rng.shuffle(label_indexes[i])
        
        # distribute class indexes to each vm according to sizes matrix
        try:
            for class_idx in range(len(data.classes)):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx
        except AttributeError:
            for class_idx in range(class_num):
                begin_idx = 0
                for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                    end_idx = begin_idx + round(frac * class_len[class_idx])
                    end_idx = int(end_idx)
                    self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                    begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]

        return selected_idxs
    
    def __len__(self):
        return len(self.data)
    
def non_iid_partition(ratio, train_class_num, initial_workers):
    partition_sizes = np.ones((train_class_num, initial_workers)) * ((1 - ratio) / (initial_workers-1))

    for i in range(train_class_num):
        partition_sizes[i][i%initial_workers]=ratio

    return partition_sizes

def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True, num_workers=0, drop_last=False, collate_fn=None):
    if selected_idxs == None:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = DataLoader(partition, batch_size=batch_size,
                                    shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
    
    return DataLoaderHelper(dataloader)


def load_datasets(dataset_type, data_path="./data/"):
    
    train_transform = load_default_transform(dataset_type, train=True)
    test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'cifar10':
        train_dataset = datasets.CIFAR10(data_path, train = True, 
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, 
                                            download = True, transform=test_transform)

    elif dataset_type == 'cifar100':
        train_dataset = datasets.CIFAR100(data_path, train = True,
                                            download = True, transform=train_transform)
        test_dataset = datasets.CIFAR100(data_path, train = False, 
                                            download = True, transform=test_transform)
    
    elif dataset_type == 'SVHN':
        train_dataset = datasets.SVHN(data_path+'/SVHN_data', split='train',
                                            download = True, transform=train_transform)
        test_dataset = datasets.SVHN(data_path+'/SVHN_data', split='test', 
                                            download = True, transform=train_transform)
    elif dataset_type == 'tinyImageNet':
        train_dataset = datasets.ImageFolder(data_path+'tiny-imagenet-200/train', transform = train_transform)
        test_dataset = datasets.ImageFolder(data_path+'tiny-imagenet-200/val/', transform = train_transform)

    return train_dataset, test_dataset


def partition_data(dataset_type, data_pattern, initial_workers=10):
    train_dataset, test_dataset = load_datasets(dataset_type)
    print(f"Total training dataset size: {len(train_dataset)}")
    labels = None
    if dataset_type in ["cifar10", 'SVHN']:
        train_class_num = 10
    elif dataset_type in ["cifar100"]:
        train_class_num = 100
    elif dataset_type == 'tinyImageNet':
        train_class_num = 200

    if data_pattern == 0:
        partition_sizes = np.ones((train_class_num, initial_workers)) * (1.0 / initial_workers)
    elif data_pattern == 1:
        non_iid_ratio = 0.2
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == 2:
        non_iid_ratio = 0.4
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == 3:
        non_iid_ratio = 0.6
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == 4:
        non_iid_ratio = 0.8
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == 5:
        non_iid_ratio = 0.9
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, initial_workers)
    elif data_pattern == -1: # this is for sharding
        partition_sizes = np.zeros((train_class_num, initial_workers))
        non_iid_ratio = 0.8
        for c in range(initial_workers):
            for l in range(train_class_num):
                flag = False

                if c%10 == l or (c+1)%10 == l:
                    flag = True

                if flag:
                    partition_sizes[l][c] = non_iid_ratio / 2
                else:
                    partition_sizes[l][c] = (1 - non_iid_ratio) / (initial_workers-1)

    
    print(partition_sizes)
    train_data_partition = LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels

############# USING MERGESFL PART END #############

def get_data_partition(dataset_type, classif_target, num_clients, shard_size, num_shards_per_partition, alpha, partition_method, batch_size=64):
    partition_method_apply = 'flower'
    if dataset_type == 'cifar10':
        partition_target = 'label'
        if partition_method == 'dl':
            partition_method_apply = 'mergesfl'
        elif partition_method == 'sharding':
            partition_method_apply = 'mergesfl'
            alpha = -1
        else:
            dataset_name = 'uoft-cs/cifar10'
    elif dataset_type == 'svhn':
        partition_method_apply = 'mergesfl'
    elif dataset_type == 'cifar100':
        dataset_name = 'uoft-cs/cifar100'
        if classif_target == 'coarse_label':
            partition_target = 'coarse_label'
            if partition_method == 'dl':
                partition_method = 'distribution'
        else:
            if num_clients == 20:
                partition_method_apply = 'flower'
            else:
                partition_method_apply = 'mergesfl'
    elif dataset_type == 'tinyImageNet':
        dataset_name = 'slegroux/tiny-imagenet-200-clean'
        partition_method_apply = 'flower'

    if partition_method_apply == 'mergesfl':
        train_dataset, test_dataset, train_data_partition, labels = partition_data(dataset_type, alpha, num_clients)
        # Clients data loaders
        bsz_list = np.ones(num_clients, dtype=int) * batch_size
        trainloaders = []
        for worker_idx in range(num_clients):
            print(f'for worker {worker_idx}')
            trainloaders.append(create_dataloaders(train_dataset, batch_size=int(bsz_list[worker_idx]), selected_idxs=train_data_partition.use(worker_idx), pin_memory=False, drop_last=True))

        testloader = create_dataloaders(test_dataset, batch_size=64, shuffle=False)
        valloaders = [[] for i in range(num_clients)]
    else:
        partition_args = {
        'dataset_name': dataset_name, ## Change this datasets for any dataset name available in https://huggingface.co/datasets.
        'num_clients': num_clients,
        'partition_method': partition_method,
        'partition_target': partition_target,
        'alpha': alpha, 
        'shard_size' : shard_size,
        'num_shards_per_partition' : num_shards_per_partition
        }
        
        trainloaders, valloaders, testloader, df_list = execute_partition_and_plot(partition_args, dataset_type, batch_size) ## there is a single testloader

    return trainloaders, valloaders, testloader, partition_method_apply