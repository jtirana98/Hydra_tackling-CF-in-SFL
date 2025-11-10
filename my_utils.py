import numpy as np
import torch

data_sets_attributes = {
    'cifar10': {'label':10},
    'cifar100': {'coarse_label':20,
                 'fine_label':100,
                 'label': 100},
    'SVHN': {'label':10},
    'tinyImageNet' : {'label':200},
}


def performance_gap(perlabel_score, ll=10):
    forgetting_ = 0

    # for e in range(100):
    for i in range(ll):
        max_val = -1000 # just a small number
        for j in range(ll):
            val = abs(min(0, perlabel_score[i] - perlabel_score[j]))
            if val > max_val:
                max_val = val
        forgetting_ += max_val
    forgetting_ = (forgetting_ / ll)
    return forgetting_

def test(net_a, net_b, testloader, dataset_name, partition_method='iid', label_name='label'):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    net_a.eval()
    if net_b != None:
        net_b.eval()

    net_a.to('cpu')
    if net_b != None:
        net_b.to('cpu')
    with torch.no_grad():
        if partition_method in ['mergesfl', 'special']:
            testloader = testloader.loader
        for batch in testloader:
            if partition_method in ['mergesfl', 'special']:
                images, labels = batch
            else:
                key_ = 'image'
                label_ = 'label'
                if dataset_name == 'cifar10':
                    key_ = 'img'
                    label_ = 'label'
                    images, labels = batch[key_], batch[label_]
                elif dataset_name == 'cifar100':
                    label_ = label_name
                    key_ = 'img'
                    images, labels = batch[key_], batch[label_]
                else:
                    images, labels = batch[key_], batch[label_]
            
            
            out_a = net_a(images)
            if net_b != None:
                outputs = net_b(out_a)
            else:
                outputs = out_a

            loss += criterion(outputs, labels).item()
            total += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def test_per_target(net_a, net_b, testloader, dataset_name, 
                    target=('label',10), partition_method='iid', 
                    label_name='label', logger=None, conf_matrix=False):

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net_a.eval()
    if net_b != None:
        net_b.eval()
    scores_accuracy = [0 for i in range(target[1])]
    total = [0 for i in range(target[1])]
    correct = [0 for i in range(target[1])]
    correct_global = 0
    total_global = 0 

    with torch.no_grad():
        if partition_method in ['mergesfl', 'special']:
            testloader = testloader.loader
        for batch in testloader:
            if partition_method in ['mergesfl', 'special']:
                images, labels = batch
                targets = labels
            else:
                key_ = 'image'
                if dataset_name == 'cifar10':
                    key_ = 'img'
                    label_ = 'label'
                    target_ = target[0]
                elif dataset_name == 'cifar100':
                    label_ = label_name
                    key_ = 'img'
                    target_ = target[0]
                else:
                    label_ = 'label'
                    key_ = 'image'
                    target_ = target[0]
                    
                images, labels, targets = batch[key_], batch[label_], batch[target_]

            out_a = net_a(images)
            if net_b != None:
                outputs = net_b(out_a)
            else:
                outputs = out_a
            
            total_global += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct_global += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                correct[targets[i]] += (predicted[i] == labels[i]).sum().item()
                total[targets[i]] += 1

    for i in range(target[1]):
        scores_accuracy[i] = correct[i] / total[i]
    
    forgetting_score = performance_gap(scores_accuracy, target[1])
    return (forgetting_score, scores_accuracy)


def test_hydra(net_a, net_b, net_c, testloader, dataset_name, partition_method='iid', label_name='label'):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net_a.eval()
    if net_b != None:
        net_b.eval()

    net_a.to('cpu')
    if net_b != None:
        net_b.to('cpu')
        net_c.to('cpu')
    with torch.no_grad():
        if partition_method == 'mergesfl':
            testloader = testloader.loader
        for batch in testloader:
            if partition_method == 'mergesfl':
                images, labels = batch
            else:
                key_ = 'image'
                if dataset_name == 'cifar10':
                    key_ = 'img'
                    label_ = 'label'
                    images, labels = batch[key_], batch[label_]
                if dataset_name == 'cifar100':
                    label_ = label_name
                    key_ = 'img'
                    images, labels = batch[key_], batch[label_]
                else:
                    label_ = label_name
                    images, labels = batch[key_], batch[label_]

            out_a = net_a(images)
            if net_b != None:
                out_b = net_b(out_a)
                outputs = net_c(out_b)
            else:
                outputs = out_a

            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def test_per_target_hydra(net_a, net_b, net_c, testloader, dataset_name, 
                    target=('label',10), partition_method='iid', 
                    label_name='label', logger=None):

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net_a.eval()
    if net_b != None:
        net_b.eval()
        net_c.eval()

    scores_accuracy = [0 for i in range(target[1])]
    total = [0 for i in range(target[1])]
    correct = [0 for i in range(target[1])]

    with torch.no_grad():
        if partition_method == 'mergesfl':
            testloader = testloader.loader
        for batch in testloader:
            if partition_method == 'mergesfl':
                images, labels = batch
                targets = labels
            else:
                key_ = 'image'
                if dataset_name == 'cifar10':
                    key_ = 'img'
                    label_ = 'label'
                    target_ = target[0]
                if dataset_name == 'cifar100':
                    label_ = label_name
                    key_ = 'img'
                    target_ = target[0]
                else:
                    label_ = label_name
                    images, labels = batch[key_], batch[label_]
                    target_ = target[0]
                images, labels, targets = batch[key_], batch[label_], batch[target_]

            out_a = net_a(images)
            if net_b != None:
                out_b = net_b(out_a)
                outputs = net_c(out_b)
            else:
                outputs = out_a

            _, predicted = torch.max(outputs.data, 1)
            for i in range(labels.size(0)):
                correct[targets[i]] += (predicted[i] == labels[i]).sum().item()
                total[targets[i]] += 1
    for i in range(target[1]):
        if total[i] == 0:
            scores_accuracy[i] = -1
        else:
            scores_accuracy[i] = correct[i] / total[i]
    
    forgetting_score = performance_gap(scores_accuracy, target[1])
    return (forgetting_score, scores_accuracy)

def test_per_target_fl3(net_a, net_b, testloader, dataset_name, 
                    target=('label',10), partition_method='iid', 
                    label_name='label', logger=None, conf_matrix=False):
    
    for i in range(target[1]):
        net_a[i].model.eval()
        net_a[i].model.to('cpu')

    if net_b != None:
        net_b.eval()
    scores_accuracy = [0 for i in range(target[1])]
    total = [0 for i in range(target[1])]
    correct = [0 for i in range(target[1])]
    correct_global = 0
    total_global = 0 

    with torch.no_grad():
        if partition_method in ['mergesfl', 'special']:
            testloader = testloader.loader
        for batch in testloader:
            if partition_method in ['mergesfl', 'special']:
                images, labels = batch
                targets = labels
            else:
                key_ = 'image'
                if dataset_name == 'cifar10':
                    key_ = 'img'
                    label_ = 'label'
                    target_ = target[0]
                elif dataset_name == 'cifar100':
                    label_ = label_name
                    key_ = 'img'
                    target_ = target[0]
                else:
                    images, labels = batch[key_], batch[label_]
                    target_ = target[0]
                images, labels, targets = batch[key_], batch[label_], batch[target_]

            predicted_ = []
            out_data_max_ = []
            for i in range(target[1]):
                out_a = net_a[i].model(images)
                if net_b != None:
                    outputs = net_b(out_a)
                else:
                    outputs = out_a
                out_data_max, predicted = torch.max(outputs.data, 1)
                predicted_.append(predicted)
                out_data_max_.append(out_data_max)
            
            total_global += labels.size(0)
            for i in range(labels.size(0)):
                correct_i = np.argmax([out_data_max_[k][i] for k in range(target[1])])

                correct_global += (predicted_[correct_i][i] == labels[i]).sum().item()
                correct[targets[i]] += (predicted_[correct_i][i] == labels[i]).sum().item()
                total[targets[i]] += 1

    for i in range(target[1]):
        if total[i] == 0:
            scores_accuracy[i] = -1
        else:
            scores_accuracy[i] = correct[i] / total[i]

    forgetting_score = performance_gap(scores_accuracy, target[1])
    return (forgetting_score, scores_accuracy, (correct_global/total_global))

def test_per_target_multihead(net_a, net_b, testloader, dataset_name, 
                    target=('label',10), partition_method='iid', 
                    label_name='label', logger=None, conf_matrix=False):
    
    for i in range(target[1]):
        net_b[i].eval()
        net_b[i].to('cpu')

    
    net_a.eval()
    scores_accuracy = [0 for i in range(target[1])]
    total = [0 for i in range(target[1])]
    correct = [0 for i in range(target[1])]
    correct_global = 0
    total_global = 0 

    with torch.no_grad():
        if partition_method in ['mergesfl', 'special']:
            testloader = testloader.loader
        for batch in testloader:
            if partition_method in ['mergesfl', 'special']:
                images, labels = batch
                targets = labels
            else:
                key_ = 'image'
                if dataset_name == 'cifar10':
                    key_ = 'img'
                    label_ = 'label'
                    target_ = target[0]
                elif dataset_name == 'cifar100':
                    label_ = label_name
                    key_ = 'img'
                    target_ = target[0]
                else:
                    images, labels = batch[key_], batch[label_]
                    target_ = target[0]
                images, labels, targets = batch[key_], batch[label_], batch[target_]

            predicted_ = []
            out_data_max_ = []
            out_a = net_a(images)
            for i in range(target[1]):
                outputs = net_b[i](out_a)
               
                out_data_max, predicted = torch.max(outputs.data, 1)
                predicted_.append(predicted)
                out_data_max_.append(out_data_max)
            
            total_global += labels.size(0)
            for i in range(labels.size(0)):
                correct_i = np.argmax([out_data_max_[k][i] for k in range(target[1])])

                correct_global += (predicted_[correct_i][i] == labels[i]).sum().item()
                correct[targets[i]] += (predicted_[correct_i][i] == labels[i]).sum().item()
                total[targets[i]] += 1

    for i in range(target[1]):
        if total[i] == 0:
            scores_accuracy[i] = -1
        else:
            scores_accuracy[i] = correct[i] / total[i]
    forgetting_score = performance_gap(scores_accuracy, target[1])
    return (forgetting_score, scores_accuracy, (correct_global/total_global))
