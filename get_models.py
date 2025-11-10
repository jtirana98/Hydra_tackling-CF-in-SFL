from models.mobilenet import MobileNetV1
from models.resnet import get_resnet_split


def get_model(model_name, dataset, cut, num_clients, is_fl=False, is_fl3=False, other=None):
    if model_name == 'mobilenet':
        if dataset in ['cifar10', 'SVHN']:
            input_dim=3
            output_dim=10
        elif dataset in 'cifar100':
            input_dim=3
            output_dim=20
            if other != 'coarse_label':
                output_dim = 100
        elif dataset in ['tinyImageNet']:
            input_dim=3
            output_dim=200
        
        if is_fl:
            cut = -1
            model_part_global = MobileNetV1(input_dim, output_dim, -1, cut)
        else:
            model_part_a = MobileNetV1(input_dim, output_dim, -1, cut)
            model_part_b = MobileNetV1(input_dim, output_dim, cut, -1)

        nets = []
        for i in range(num_clients):
            nets.append(MobileNetV1(input_dim, output_dim, -1, cut))
        if is_fl3:
            nets_b = []
            for i in range(num_clients):
                nets_b.append(MobileNetV1(input_dim, output_dim, cut, -1))
    elif 'resnet' in model_name:
        output_dim = 10
        if is_fl:
            cut = -1
            if dataset in ['cifar10', 'SVHN']:
                input_dim=3
                output_dim=10
            elif dataset in ['cifar100']:
                input_dim=3
                output_dim=20
                if other != 'coarse_label':
                    output_dim = 100
            elif dataset in ['tinyImageNet']:
                input_dim=3
                output_dim=200
            else:
                inputs = 3
            model_part_global =  get_resnet_split(inputs, output_dim, -1, -1, model_name)
        else:
            if dataset in ['cifar10', 'SVHN']:
                input_dim=3
                output_dim=10
            elif dataset in ['cifar100']:
                input_dim=3
                output_dim=20
                if other != 'coarse_label':
                    output_dim = 100
            elif dataset in ['tinyImageNet']:
                input_dim=3
                output_dim=200
            else:
                inputs = 3
            input_dim=3
            model_part_a = get_resnet_split(input_dim, output_dim, -1, cut, model_name)
            model_part_b = get_resnet_split(input_dim, output_dim, cut, -1, model_name)
        nets = []
        for i in range(num_clients):
            nets.append(get_resnet_split(input_dim, output_dim, -1, cut, model_name))
        if is_fl3:
            nets_b = []
            for i in range(num_clients):
                nets_b.append(get_resnet_split(input_dim, output_dim, cut, -1, model_name))
    if is_fl:
        return model_part_global, nets
    elif is_fl3:
        return (model_part_a, model_part_b), nets, nets_b
    else:
        return (model_part_a, model_part_b), nets

def get_model_threesplit(model_name, dataset, cut_a, cut_b, num_clients, other=None):
    if model_name == 'mobilenet':
        if dataset in ['cifar10', 'SVHN']:
            input_dim=3
            output_dim=10
        elif dataset in ['cifar100']:
            input_dim=3
            output_dim=20
            if other != 'coarse_label':
                output_dim = 100
        elif dataset in ['tinyImageNet']:
            input_dim=3
            output_dim=200
        else:
            input_dim=3
        
        model_part_a = MobileNetV1(input_dim, output_dim, -1, cut_a)
        model_part_b = MobileNetV1(input_dim, output_dim, cut_a, cut_b)
        model_part_c = MobileNetV1(input_dim, output_dim, cut_b, -1)

        nets_a = [] 
        nets_c = []
        for i in range(num_clients):
            nets_a.append(MobileNetV1(input_dim, output_dim, -1, cut_a))
            nets_c.append(MobileNetV1(input_dim, output_dim, cut_b, -1))
    elif 'resnet' in model_name:
        if dataset in ['cifar10', 'SVHN']:
            input_dim=3
            output_dim=10
        elif dataset in ['cifar100']:
            input_dim=3
            output_dim=20
            if other != 'coarse_label':
                output_dim = 100
        elif dataset in ['tinyImageNet']:
            input_dim=3
            output_dim=200
        else:
            input_dim=3
            
        model_part_a = get_resnet_split(input_dim, output_dim, -1, cut_a, model_name)
        model_part_b = get_resnet_split(input_dim, output_dim, cut_a, cut_b, model_name)
        model_part_c = get_resnet_split(input_dim, output_dim, cut_b, -1, model_name)

        nets_a = []
        nets_c = []
        for i in range(num_clients):
            nets_a.append(get_resnet_split(input_dim, output_dim, -1, cut_a, model_name))
            nets_c.append(get_resnet_split(input_dim, output_dim, cut_b, -1, model_name))

    return (model_part_a, model_part_b, model_part_c), (nets_a, nets_c)