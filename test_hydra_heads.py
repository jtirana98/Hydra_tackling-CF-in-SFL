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

teams_5 = {
    0: 0,
    1: 1,
    2: 0,
    3: 3,
    4: 4,
    5: 3,
    6: 4,
    7: 3,
    8: 2,
    9: 1,
}

# 0 animal, 1 no-animal
teams_2 = {
    0: 1,
    1: 1,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 1,
    9: 1,
}

teams = {
    2: teams_2,
    5: teams_5
}


#init parameters
def init_parameters():
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset_type', type=str, default='cifar10', help='cifar10/cifar100/svhn/tinyImageNet')
    parser.add_argument('--model_name', type=str, default='mobilenet', help='e.g., mobilenet/resnet18/resnet101')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--cut_a', type=int, default=3)
    parser.add_argument('--cut_b', type=int, default=3)
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
    parser.add_argument('--num_heads', type=int, default=5, help='number of groups--heads')
    parser.add_argument('--policy', type=str, default='cyclic', help='server processing order policy: cyclic/reverse/random')
    parser.add_argument('--phi', type=int, default=10, help='\phi parameter that denote the multiplier parameter for the number of clients')
    args = parser.parse_args()

    return args

class Model_c:
    def __init__(self, id, model_c, optimizer_c):
        self.model_c = model_c
        self.optimizer_c = optimizer_c

    def next_job(self, det_out_b, labels):
        self.model_c.train()

        criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_c.zero_grad()

        self.model_c.to(device)
        out = self.model_c(det_out_b)
        labels = labels.long()

        loss = criterion(out, labels)
        loss.backward()

        self.optimizer_c.step()
        grad_b = det_out_b.grad.clone().detach()
        grad_b.to(device)

        return(grad_b, det_out_b)

class Client_usplit:
    def __init__(self, id, model_a, model_c_instance, optimizer_a, dataloader, valloader, total_epoch, dataset_name, partition_method, classif_target=None):
        self.id = id

        self.model_a = model_a
        self.optimizer_a = optimizer_a

        self.model_c_instance = model_c_instance


        self.dataset_name = dataset_name
        
        self.dataloader = dataloader
        self.valloader = valloader

        self.partition_method  = partition_method
        self.local_epoch_count = 0
        self.batch_count = 0
        self.prev_job = 1 # 0: for forward and 1: for backward

        self.total_batch = len(self.dataloader)
        self.total_epoch = total_epoch
        print(self.total_batch)
        # other status parameters
        self.labels = None
        self.my_outa = None
        self.det_out_a = None
        self.grad_a = None
        
        self.correct = 0
        self.total = 0
        self.epoch_loss = 0.0

        self.end = False
        self.myiter = iter(self.dataloader)

        self.start_init = False
        self.classif_target = classif_target 
    
    def start_a(self):
        self.myiter = iter(self.dataloader)
        self.total_batch = len(self.dataloader)

    def next_job(self):
        self.model_a.train()
        
        self.prev_job = (self.prev_job + 1) % 2
        
        job = {
            'id' : self.id,
            'type' : self.prev_job
        }

        
        if self.prev_job == 0:
            if self.partition_method == 'mergesfl':
                inputs, labels = next(self.myiter)
            else:
                batch = next(self.myiter)
                key_ = 'image'
                if args.dataset_type == 'cifar10':
                    key_ = 'img'
                    label_ = 'label'
                if args.dataset_type == 'cifar100':
                    label_  = self.classif_target 
                    key_ = 'img'
                elif self.dataset_name == 'tinyImageNet':
                    key_ = 'image'
                    label_ = 'label'
                inputs, labels = batch[key_], batch[label_]
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer_a.zero_grad()
            self.labels = labels

            self.model_a.to(device)
            self.my_outa = self.model_a(inputs)
            self.my_outa.requires_grad_(True)
            self.det_out_a = self.my_outa.clone().detach().requires_grad_(True)
            self.det_out_a.to(device)
        else:
            self.optimizer_a.zero_grad()
            self.my_outa.backward(self.grad_a) # take gradients from helper
            self.optimizer_a.step()

            self.batch_count = (self.batch_count + 1) % self.total_batch
    
            if self.batch_count == 0: # end of the epoch
                self.local_epoch_count += 1
                self.correct = 0
                self.total = 0
                self.epoch_loss = 0.0
                self.myiter = iter(self.dataloader)
                if self.local_epoch_count >= self.total_epoch:
                    self.end = True
                    self.local_epoch_count = 0

        return job
    

    def next_job_c(self):
        self.grad_b, self.det_out_b = self.model_c_instance.next_job(self.det_out_b, self.labels)

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
    (global_model_a, global_model_b, global_model_c), (nets_a, nets_c) = get_models.get_model_threesplit(args.model_name, args.dataset_type, args.cut_a, args.cut_b,  args.num_clients, other=args.classif_target)
    (_, global_model_bb, _), _ = get_models.get_model_threesplit(args.model_name, args.dataset_type, args.cut_a, args.cut_b,  1, other=args.classif_target)
    global_model_a_par = global_model_a.state_dict()
    global_model_c_par = global_model_c.state_dict()
        
    for client in range(args.num_clients):
        nets_a[client].load_state_dict(global_model_a_par)
        nets_c[client].load_state_dict(global_model_c_par)
        
    client_optimizers_a = []
    client_optimizers_c = []
    if args.momentum < 0:
        global_optim = optim.SGD(global_model_b.parameters(), lr=args.training_lr, weight_decay=args.weight_decay)
        for client in range(args.num_clients):
            client_optimizers_a.append(optim.SGD(nets_a[client].parameters(), lr=args.training_lr, weight_decay=args.weight_decay))
            client_optimizers_c.append(optim.SGD(nets_c[client].parameters(), lr=args.training_lr, weight_decay=args.weight_decay))
    else:
        global_optim = optim.SGD(global_model_b.parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
        for clients in range(args.num_clients):
            client_optimizers_a.append(optim.SGD(nets_a[client].parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))
            client_optimizers_c.append(optim.SGD(nets_c[client].parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))

    label_ = args.classif_target
    trainloaders, valloaders, testloader, partition_method_apply = get_data.get_data_partition(args.dataset_type, args.classif_target, args.num_clients, args.shard_size, args.num_shards_per_partition, args.alpha, args.partition_method, args.batch_size)


    if args.policy in ['cyclic', 'reverse']:
        artificial_order = []
        while len(artificial_order) < args.num_clients:
            pick = random.randint(0, args.num_clients-1)
            if pick not in artificial_order:
                artificial_order.append(pick)

        if args.dataset_type == 'SVHN':
            artificial_order = [9, 0, 8, 7, 6, 5, 4, 3, 2, 1]   # sorting them by increasing order of data samples per label for a fairer study

    clients_state = []
    job_buffer = []

    # initialize buffer of jobs        
    model_c_instances = []
    for g in range(args.num_heads):
        model_c_instances.append(Model_c(g, nets_c[g], client_optimizers_c[g]))

    for client in range(args.num_clients):
        clients_state.append(Client_usplit(client, nets_a[client], model_c_instances[teams[args.num_heads][client]],
                                     client_optimizers_a[client],
                                     trainloaders[client], valloaders[client], 
                                     args.local_epoch, dataset_name=args.dataset_type, partition_method=partition_method_apply, classif_target=args.classif_target))        

    epoch_lr = args.training_lr
    lambda_reg = 1e-4
    count_f = 0
    for epoch in range(args.epoch):
        if epoch >= 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
            lambda_reg = lambda_reg*args.decay_rate
            if args.momentum < 0:
                global_optim = optim.SGD(global_model_b.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
                for client in range(args.num_clients):
                    clients_state[client].optimizer_a = optim.SGD(clients_state[client].model_a.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
                    if client < args.num_heads:
                        model_c_instances[client].optimizer_c = optim.SGD(model_c_instances[client].model_c.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
            else:
                global_optim = optim.SGD(global_model_b.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
                for clients in range(args.num_clients):
                    clients_state[client].optimizer_a = optim.SGD(clients_state[client].model_a.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
                    if client < args.num_heads:
                        model_c_instances[client].optimizer_c = optim.SGD(model_c_instances[client].model_c.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        for client in range(args.num_clients):
            if args.policy in ['cyclic', 'reverse']:
                the_client = artificial_order[client]
            else:
                the_client = client
            
            clients_state[the_client].end = False
            clients_state[the_client].start_a()
            clients_state[the_client].model_a.to(device)
            
            job = clients_state[the_client].next_job()
            job_buffer.append(job)

        for client in range(args.num_heads):
            model_c_instances[client].model_c.to(device)

        global_model_b.to(device)

        next_index_order = 0
        order_ = 0
        while len(job_buffer) > 0:  
            if args.policy in ['cyclic']:
                next_index = 0
            elif args.policy == 'reverse':
                next_index = -next_index_order
            elif args.policy == 'random':
                next_index = random.randint(0, len(job_buffer)-1)

            job = job_buffer[next_index]
            del job_buffer[next_index]
            
            global_optim.zero_grad()

            the_client = job['id']
            
            out_b = global_model_b(clients_state[the_client].det_out_a)

            clients_state[the_client].det_out_b = out_b.clone().detach().requires_grad_(True)
            clients_state[the_client].next_job_c()
            out_b.backward(clients_state[the_client].grad_b)

            global_optim.step()
            grad_a = clients_state[the_client].det_out_a.grad.clone().detach()
            grad_a.to(device)

            clients_state[the_client].grad_a = grad_a
            clients_state[the_client].next_job() # client completes backward

            if not clients_state[the_client].end:
                job = clients_state[the_client].next_job()
                if args.policy in ['cyclic', 'reverse']:
                    if count_f < args.phi-1:
                        if args.policy == 'cyclic':
                            job_buffer.insert(0,job)
                        elif args.policy == 'reverse':
                            if next_index_order == 0:
                                job_buffer.insert(0,job)
                            else:
                                job_buffer.append(job)
                        count_f += 1
                    else:
                        count_f = 0
                        if args.policy == 'cyclic':
                            job_buffer.append(job)
                        elif args.policy == 'reverse':
                            if next_index_order == 0:
                                job_buffer.append(job)
                            else:
                                job_buffer.insert(0,job)
                        order_ = (order_+1)%len(artificial_order)
                        if order_ == 0:
                            next_index_order = (next_index_order+1) % 2
                else:
                    job_buffer.append(job)
            else:
                count_f = 0
                if args.policy != 'random':
                    order_ = (order_+1)%len(artificial_order)

        # aggregation
        with torch.no_grad():
            # Aggregate client models
            for the_client in range(args.num_clients):
                clients_state[the_client].model_a.to('cpu')
                net_para = clients_state[the_client].model_a.cpu().state_dict()
                
                if the_client == 0:
                    for key in net_para:
                        global_model_a_par[key] = net_para[key] / args.num_clients
                else:
                    for key in net_para:
                        global_model_a_par[key] += net_para[key] / args.num_clients
        
        with torch.no_grad():
            # Aggregate client models
            for the_client in range(args.num_heads):
                model_c_instances[the_client].model_c.to('cpu')
                net_para = model_c_instances[the_client].model_c.cpu().state_dict()
                
                if the_client == 0:
                    for key in net_para:
                        global_model_c_par[key] = net_para[key] / args.num_heads
                else:
                    for key in net_para:
                        global_model_c_par[key] += net_para[key] / args.num_heads


        global_model_a.load_state_dict(global_model_a_par)
        global_model_c.load_state_dict(global_model_c_par)
        for the_client in range(args.num_clients):
            clients_state[the_client].model_a.load_state_dict(global_model_a_par)
            
            if the_client < args.num_heads:
                model_c_instances[the_client].model_c.load_state_dict(global_model_c_par)

       
        # Epoch ecaluation
        global_model_b_par = global_model_b.state_dict()
        global_model_bb.load_state_dict(global_model_b_par)

        epoch_loss, epoch_accuracy = my_utils.test_hydra(global_model_a, global_model_bb, global_model_c, testloader, args.classif_target, partition_method_apply, label_)
        logger.info(f'>>>>> Epoch [{epoch}]: Global model loss: {epoch_loss} acc: {epoch_accuracy}')
        

        (forg_score, acc_scores) = my_utils.test_per_target_hydra(global_model_a, global_model_bb, global_model_c, testloader, args.dataset_type,
                                                target=(args.classif_target, my_utils.data_sets_attributes[args.dataset_type][args.classif_target]),
                                                partition_method=partition_method_apply, label_name=label_)
        
        logger.info(f'>>>>> Forgetting score: {forg_score}')
        if args.policy in ['cyclic', 'reverse']:
            logger.info(f'Scores per-position value')
        else:
            logger.info(f'Scores per-label value')
        
        for i in range(len(acc_scores)):
            if args.policy in ['cyclic', 'reverse']:
                logger.info(f'position {i} >> {acc_scores[artificial_order[i]]}')
            else:
                logger.info(f'label {i} >> {acc_scores[i]}')


if __name__ == '__main__':
    main()