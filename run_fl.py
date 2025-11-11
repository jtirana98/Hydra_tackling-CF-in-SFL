import argparse
import random
import logging
import datetime
import time
import os

import torch
import torch.optim as optim

import get_models
import get_data
import my_utils

args = None
device = None


#init parameters
def init_parameters():
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset_type', type=str, default='cifar10', help='cifar10/cifar100/svhn/tinyImageNet')
    parser.add_argument('--model_name', type=str, default='mobilenet', help='e.g., mobilenet/resnet18/resnet101')
    parser.add_argument('--num_clients', type=int, default=10) 
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--training_lr', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.993)
    parser.add_argument('--min_lr', type=float, default=0.005)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--partition_method', type=str, default='iid', help='This can be iid/dirichlet/quantity/archetype/distribution/mergesfl')
    parser.add_argument('--classif_target', type=str, default='label', help="The target according to which the data will be partioned, e.g., age', 'household_position', 'household_size', etc.")
    parser.add_argument('--alpha', type=float, default=1, help='for dirichlet')
    parser.add_argument('--num_shards_per_partition', type=int, default=2, help='Number of shrads each client will have in archetype')
    parser.add_argument('--shard_size', type=int, default=500, help='Shrads size in archetype')
    args = parser.parse_args()

    return args

class Client_FL:
    def __init__(self, id, model, optimizer, dataloader, valloader, total_epoch, dataset_name, device, partition_method, classif_target=None):
        self.id = id

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.dataset_name = dataset_name
        
        self.dataloader = dataloader
        self.valloader = valloader

        self.total_batch = len(self.dataloader)
        self.total_epoch = total_epoch
        print(self.total_batch)

        self.partition_method  = partition_method
        self.classif_target = classif_target 
    
    def run_epoch(self):
        self.model.train()
        self.myiter = iter(self.dataloader)
        self.epoch_loss = 0
        self.total = 0

        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        for local_epoc in range(self.total_epoch):
            for b in range(self.total_batch):
                if self.partition_method == 'mergesfl':
                    inputs, labels = next(self.myiter)
                else:
                    batch = next(self.myiter)
                    key_ = 'image'
                    if self.dataset_name == 'cifar10':
                        key_ = 'img'
                        label_ = 'label'
                    elif self.dataset_name == 'cifar100':
                        label_ = self.classif_target #'coarse_label'
                        key_ = 'img'
                    elif self.dataset_name == 'tinyImageNet':
                        key_ = 'image'
                        label_ = 'label'
                    inputs, labels = batch[key_], batch[label_]
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                self.model.to(self.device)
                out = self.model(inputs)

                labels = labels.long()

                loss = criterion(out, labels)
                loss.backward()

                self.epoch_loss += loss.item()
                self.total += labels.size(0)
                correct += (torch.max(out.data, 1)[1] == labels).sum().item()

                self.optimizer.step()
    
def main():
    global args, device

    args = init_parameters()
    device = torch.device(args.device)
    print(args.__dict__)

    logdir = 'logs_training'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    while True:
        log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
        log_path = log_file_name+'.log'
    
        if not os.path.exists(os.path.join(logdir, log_path)):
            break
        else:
            time.sleep(1)
    
    logging.basicConfig(
        filename=os.path.join(logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args.device)
    logger.info(args.__dict__)
    logger.info("#" * 100)

    # initialize models
    global_model, nets = get_models.get_model(args.model_name, args.dataset_type, -1, args.num_clients, is_fl=True)
    global_model_par = global_model.state_dict()
    for client in range(args.num_clients):
        nets[client].load_state_dict(global_model_par)
        
    client_optimizers = []
    if args.momentum < 0:
        for client in range(args.num_clients):
            client_optimizers.append(optim.SGD(nets[client].parameters(), lr=args.training_lr, weight_decay=args.weight_decay))
    else:
        for clients in range(args.num_clients):
            client_optimizers.append(optim.SGD(nets[client].parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))


    label_ = args.classif_target
    trainloaders, valloaders, testloader, partition_method_apply = get_data.get_data_partition(args.dataset_type, args.classif_target, args.num_clients, args.shard_size, args.num_shards_per_partition, args.alpha, args.partition_method, args.batch_size)
    
    clients_state = []
    job_buffer = []

    # initialize buffer of jobs
    for client in range(args.num_clients):
        clients_state.append(Client_FL(client, nets[client], client_optimizers[client],
                                     trainloaders[client], valloaders[client], 
                                     args.local_epoch,args.dataset_type, device, partition_method_apply, args.classif_target))
    epoch_lr = args.training_lr
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epoch):
        if epoch > 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))

            if args.momentum < 0:
                for client in range(args.num_clients):
                    clients_state[client].optimizer = optim.SGD(clients_state[client].model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
            else:
                for clients in range(args.num_clients):
                    clients_state[client].optimizer = optim.SGD(clients_state[client].model.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        for client in range(args.num_clients):
            clients_state[client].run_epoch()

        # aggregation
        with torch.no_grad():
            for the_client in range(args.num_clients):
                clients_state[the_client].model.to('cpu')
                net_para = clients_state[the_client].model.cpu().state_dict()
                
                if the_client == 0:
                    for key in net_para:
                        global_model_par[key] = net_para[key] / args.num_clients
                else:
                    for key in net_para:
                        global_model_par[key] += net_para[key] / args.num_clients
        
        
        global_model.load_state_dict(global_model_par)
        for the_client in range(args.num_clients):
            clients_state[the_client].model.load_state_dict(global_model_par)

        epoch_loss, epoch_accuracy = my_utils.test(global_model, None, testloader, args.dataset_type, partition_method_apply, label_)
        logger.info(f'>>>>> Epoch [{epoch}]: Global model loss: {epoch_loss} acc: {epoch_accuracy}')
        

        (forg_score, acc_scores, bw_score_val) = my_utils.test_per_target(global_model, None, testloader, args.dataset_type,
                                        target=(args.classif_target, my_utils.data_sets_attributes[args.dataset_type][args.classif_target]),
                                        partition_method=partition_method_apply, label_name=label_)
        logger.info(f'>>>>> Performance Gap: {forg_score}')
        logger.info(f'>>>>> Backward Tranfer: {bw_score_val}')
        logger.info(f'Scores per-label value')
        for i in range(len(acc_scores)):    
            logger.info(f'label {i} >> {acc_scores[i]}')

if __name__ == '__main__':
    main()
