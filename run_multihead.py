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

import run_sfl


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
    args = parser.parse_args()

    return args

    
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
    (global_model_a, global_model_b), nets, nets_b = get_models.get_model(args.model_name, args.dataset_type, args.cut, args.num_clients, is_fl3=True)
    (_, global_model_bb), _ = get_models.get_model(args.model_name, args.dataset_type, args.cut, 1)
    global_model_a_par = global_model_a.state_dict()
    global_model_b_par = global_model_b.state_dict()
        
    for client in range(args.num_clients):
        nets[client].load_state_dict(global_model_a_par)
        nets_b[client].load_state_dict(global_model_b_par)
        
    client_optimizers = []
    client_optimizers_b = []
    if args.momentum < 0:
        for client in range(args.num_clients):
            client_optimizers.append(optim.SGD(nets[client].parameters(), lr=args.training_lr, weight_decay=args.weight_decay))
            client_optimizers_b.append(optim.SGD(nets_b[client].parameters(), lr=args.training_lr, weight_decay=args.weight_decay))
    else:
        for client in range(args.num_clients):
            client_optimizers.append(optim.SGD(nets[client].parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))
            client_optimizers_b.append(optim.SGD(nets_b[client].parameters(), lr=args.training_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay))


    label_ = args.classif_target
    trainloaders, valloaders, testloader, partition_method_apply = get_data.get_data_partition(args.dataset_type, args.classif_target, args.num_clients, args.shard_size, args.num_shards_per_partition, args.alpha, args.partition_method, args.batch_size)
    
    clients_state = []
    job_buffer = []

    # initialize buffer of jobs
    for client in range(args.num_clients):
        clients_state.append(run_sfl.Client(client, nets[client], client_optimizers[client],
                                     trainloaders[client], valloaders[client], 
                                     args.local_epoch,args.dataset_type, device, partition_method_apply, args.classif_target))
    epoch_lr = args.training_lr
    criterion = torch.nn.CrossEntropyLoss()
    lambda_reg = 1e-4
    count_f = 0
    for epoch in range(args.epoch):
        if epoch >= 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))

            if args.momentum < 0:
                for client in range(args.num_clients):
                    clients_state[client].optimizer = optim.SGD(clients_state[client].model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
                    client_optimizers_b[client] = optim.SGD(nets_b[client].parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
            else:
                for client in range(args.num_clients):
                    clients_state[client].optimizer = optim.SGD(clients_state[client].model.parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
                    client_optimizers_b[client] = optim.SGD(nets_b[client].parameters(), lr=epoch_lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        for client in range(args.num_clients):
            the_client = client
            
            clients_state[the_client].end = False
            clients_state[the_client].start()
            clients_state[the_client].model.to(device)
            
            job = clients_state[the_client].next_job()
            job_buffer.append(job)

        global_model_b.to(device)

        next_index_order = 0
        order_ = 0
        while len(job_buffer) > 0:  
            next_index = 0

            job = job_buffer[next_index]
            del job_buffer[next_index]
            
            client_optimizers_b[the_client].zero_grad()
            the_client = job['id']
            nets_b[the_client].to(device)
            out = nets_b[the_client](clients_state[the_client].det_out_a)
            labels = clients_state[the_client].labels.long()
            
            loss = criterion(out, labels)
            loss.backward()
            clients_state[the_client].epoch_loss += loss.item()
            clients_state[the_client].total += labels.size(0)
            
            clients_state[the_client].correct += (torch.max(out.data, 1)[1] == labels).sum().item()

            client_optimizers_b[the_client].step()
            grad_a = clients_state[the_client].det_out_a.grad.clone().detach()
            grad_a.to(device)
            
            clients_state[the_client].grad_a = grad_a

            clients_state[the_client].next_job() # client completes backward
            if not clients_state[the_client].end:
                job = clients_state[the_client].next_job() # client start next job
                job_buffer.insert(0, job)

        # aggregation
        with torch.no_grad():
            # Aggregate client models
            for the_client in range(args.num_clients):
                clients_state[the_client].model.to('cpu')
                net_para = clients_state[the_client].model.cpu().state_dict()
                
                if the_client == 0:
                    for key in net_para:
                        global_model_a_par[key] = net_para[key] / args.num_clients
                else:
                    for key in net_para:
                        global_model_a_par[key] += net_para[key] / args.num_clients
        

        global_model_a.load_state_dict(global_model_a_par)
        for the_client in range(args.num_clients):
            clients_state[the_client].model.load_state_dict(global_model_a_par)


        # Epoch ecaluation
        global_model_b_par = global_model_b.state_dict()
        global_model_bb.load_state_dict(global_model_b_par)
         
        forg_score, scores, epoch_accuracy = my_utils.test_per_target_multihead(global_model_a, nets_b, testloader, args.dataset_type,
                                                target=(args.classif_target, my_utils.data_sets_attributes[args.dataset_type][args.classif_target]),
                                                partition_method=partition_method_apply, label_name=label_)
       
        logger.info(f'>>>>> Epoch [{epoch}]: Global model acc: {epoch_accuracy}')
        logger.info(f'>>>>> Performance Gap: {forg_score}')
        logger.info(f'Scores per target value')
        for i in range(my_utils.data_sets_attributes[args.dataset_type][args.classif_target]):
            logger.info(f'for label {i} >> {scores[i]}')      

if __name__ == '__main__':
    main()
