import argparse
import numpy as np
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
    parser.add_argument('--cut', type=int, default=3)
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
    parser.add_argument('--phi', type=int, default=10, help='\phi parameter that denote the multiplier parameter for the number of clients')
    args = parser.parse_args()

    return args

class Client_mergesfl:
    def __init__(self, id, model, optimizer, dataloader, valloader, total_epoch, dataset_name, device, phi_par, partition_method, classif_target=None):
        self.id = id
        self.model_id = 0

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.dataset_name = dataset_name
        self.partition_method = partition_method
        self.classif_target = classif_target
        self.phi_par = phi_par
        self.dataloader = dataloader
        self.valloader = valloader

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
    
    def start(self):
        self.myiter = iter(self.dataloader)
        self.total_batch = len(self.dataloader)
        self.prev_job = 1

    def next_job(self):
        self.model[self.model_id].train()
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
                elif args.dataset_type == 'cifar100':
                    label_ = self.classif_target
                    key_ = 'img'
                inputs, labels = batch[key_], batch[label_]
                
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer[self.model_id].zero_grad()
            self.labels = labels

            self.model[self.model_id].to(self.device)
            self.my_outa = self.model[self.model_id](inputs)
            self.my_outa.requires_grad_(True)
            self.det_out_a = self.my_outa.clone().detach().requires_grad_(True)
            self.det_out_a.to(self.device)
        else:
            self.optimizer[self.model_id].zero_grad()
            self.my_outa.backward(self.grad_a) # take gradients from helper
            self.optimizer[self.model_id].step()

            self.batch_count = (self.batch_count + 1) % self.total_batch
            self.model_id = (self.model_id + 1) % self.phi_par
            if self.batch_count == 0: # end of the epoch
                self.local_epoch_count += 1
                self.correct = 0
                self.total = 0
                self.epoch_loss = 0.0
                self.myiter = iter(self.dataloader)
                if self.local_epoch_count >= self.total_epoch:
                    self.end = True
                    self.local_epoch_count = 0
                    self.model_id = 0

        return job
    
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
    (global_model_a, global_model_b), nets = get_models.get_model(args.model_name, args.dataset_type, args.cut, args.num_clients*args.phi)
    (_, global_model_bb), _ = get_models.get_model(args.model_name, args.dataset_type, args.cut, 1)
    global_model_a_par = global_model_a.state_dict()
        
    for client in range(args.num_clients*args.phi):
        nets[client].load_state_dict(global_model_a_par)
        
    client_optimizers = []
    if args.momentum < 0:
        global_optim = optim.SGD(global_model_b.parameters(), lr=args.training_lr, weight_decay=args.weight_decay)
        for client in range(args.num_clients*args.phi):
            client_optimizers.append(optim.SGD(nets[client].parameters(), lr=args.training_lr, weight_decay=args.weight_decay))
    else:
        global_optim = optim.SGD(global_model_b.parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
        for client in range(args.num_clients*args.phi):
            client_optimizers.append(optim.SGD(nets[client].parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))


    label_ = args.classif_target
    trainloaders, valloaders, testloader, partition_method_apply = get_data.get_data_partition(args.dataset_type, args.classif_target, args.num_clients, args.shard_size, args.num_shards_per_partition, args.alpha, args.partition_method, args.batch_size)
    
    clients_state = []
    job_buffer = []

    # initialize buffer of jobs
    clients_state = []
    job_buffer = []
    for client in range(args.num_clients):
        nets__ = []
        optimizers = []
        for i in range(args.phi):
            nets__.append(nets[client*args.phi+i])
            optimizers.append(client_optimizers[client*args.phi+i])
        clients_state.append(Client_mergesfl(client, nets__, optimizers,
                                     trainloaders[client], valloaders[client], 
                                     args.local_epoch, args.dataset_type, device, args.phi, partition_method_apply, args.classif_target))
        
    epoch_lr = args.training_lr
    criterion = torch.nn.CrossEntropyLoss()
    bsz_list = np.ones(args.num_clients, dtype=int) * args.batch_size
    for epoch in range(args.epoch):
        if epoch >= 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))

            if args.momentum < 0:
                global_optim = optim.SGD(global_model_b.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
                for client in range(args.num_clients):
                    for i in range(args.phi):
                        clients_state[client].optimizer[i] = optim.SGD(clients_state[client].model[i].parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
            else:
                global_optim = optim.SGD(global_model_b.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
                for client in range(args.num_clients):
                    for i in range(args.phi):
                        clients_state[client].optimizer[i] = optim.SGD(clients_state[client].model[i].parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        
        for client in range(args.num_clients):
            the_client = client
            clients_state[the_client].end = False
            clients_state[the_client].start()
            for i in range(args.phi):
                clients_state[the_client].model[i].to(device)
            
            job = clients_state[the_client].next_job()

            job_buffer.append(job)

        global_model_b.to(device)

        while len(job_buffer) > 0:  
            smash_data = []
            labels_ = []

            for next_index in range(len(job_buffer)):
                the_client = job_buffer[next_index]['id']
                labels = clients_state[the_client].labels.long()

                smash_data.append(clients_state[the_client].my_outa.detach())
                labels_.append(labels)

            m_data = torch.cat(smash_data)
            m_target = torch.cat(labels_)
            m_data = m_data.requires_grad_(True)
            
            global_optim.zero_grad()
            out = global_model_b(m_data)
            loss = criterion(out, m_target.long())
            loss.backward()
            global_optim.step()

            # gradient dispatch
            sum_bsz = sum([bsz_list[i] for i in range(len(job_buffer))])
            bsz_s = 0
            initial_len = len(job_buffer)

            for i in range(initial_len):
                worker_idx = 0
                the_client = job_buffer[worker_idx]['id']

                clients_grad = m_data.grad[bsz_s: bsz_s + bsz_list[worker_idx]] * sum_bsz / bsz_list[worker_idx]
                bsz_s += bsz_list[worker_idx]
            
                clients_grad.to(device)
                clients_state[the_client].grad_a = clients_grad

                job = job_buffer[worker_idx]
                del job_buffer[worker_idx]

                clients_state[the_client].next_job() # client completes backward
                if not clients_state[the_client].end:
                    job = clients_state[the_client].next_job() # client start next job
                    job_buffer.append(job)


        # aggregation
        with torch.no_grad():
            # Aggregate client models
            for the_client in range(args.num_clients):
                for i in range(args.phi):
                    clients_state[the_client].model[i].to('cpu')
                    net_para = clients_state[the_client].model[i].cpu().state_dict()
                
                if the_client == 0:
                    for key in net_para:
                        global_model_a_par[key] = net_para[key] / (args.num_clients)
                else:
                    for key in net_para:
                        global_model_a_par[key] += net_para[key] / (args.num_clients)
        
        
        global_model_a.load_state_dict(global_model_a_par)
        for the_client in range(args.num_clients):
            for i in range(args.phi):
                clients_state[the_client].model[i].load_state_dict(global_model_a_par)

        # Epoch ecaluation
        global_model_b_par = global_model_b.state_dict()
        global_model_bb.load_state_dict(global_model_b_par)

        epoch_loss, epoch_accuracy = my_utils.test(global_model_a, global_model_bb, testloader, args.dataset_type, partition_method_apply, label_)
        logger.info(f'>>>>> Epoch [{epoch}]: Global model loss: {epoch_loss} acc: {epoch_accuracy}')
        

        (forg_score, acc_scores) = my_utils.test_per_target(global_model_a, global_model_bb, testloader, args.dataset_type,
                                        target=(args.classif_target, my_utils.data_sets_attributes[args.dataset_type][args.classif_target]),
                                        partition_method=partition_method_apply, label_name=label_)
        logger.info(f'>>>>> Forgetting score: {forg_score}')
        logger.info(f'Scores per-label value')
        
        for i in range(len(acc_scores)):
            logger.info(f'label {i} >> {acc_scores[i]}')

if __name__ == '__main__':
    main()
