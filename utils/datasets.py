"""
Dataset loading utilities
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
import sklearn.datasets as sklearn_datasets

from torch.utils.data import TensorDataset, Subset

from utils.auto_augmentation import auto_augment_policy, AutoAugment
from utils.random_augmentation import rand_augment_transform
from utils.random_erasing import RandomErasing
from utils.aug_mix_dataset import AugMixDataset, Dataset
from utils.celeba_backdoor import BackdoorCelebA
from utils.celeba_full import FullCelebA
from utils.celeba_label_specific import LabelSpecificCelebA

from PIL import Image
import pandas as pd

import pdb

DATASETS_NAMES = ['imagenet', 'cifar10', 'cifar100', 'mnist',
                  'celeba', 'backdoorceleba', 'awa2', 'full_celeba']#, "label_specific_celeba"]

__all__ = ["get_datasets", "extract_resnet50_features", "extract_mobilenet_features", "classification_num_classes", "interpolation_flag"]



def _classification_dataset_str_from_arch(arch):
    if 'cifar100' in arch:
        dataset = 'cifar100'
    elif 'cifar' in arch:
        dataset = 'cifar10'
    elif 'mnist' in arch:
        dataset = 'mnist'
    else:
        dataset = 'imagenet'
    return dataset


def extract_resnet50_features(model, data_loader, device, update_bn_stats=False, bn_updates=10, 
                              str_model=False, madry_model=False, timm_model=False):
    
    # this is where the extracted features will be added:
    data_features_tensor = None
    data_labels_tensor = None

    # original data loader
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    
    if update_bn_stats:
        # if true, update the BatchNorm stats by doing a few dummy forward passes through the data
        # if false, use the ImageNet Batch Norm stats
        model.train()
        for i in range(bn_updates):
            with torch.no_grad():
                for sample, target in data_loader:
                    sample = sample.to(device)
                    model(sample)
    model.eval()

    #register a forward hook:
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    if not timm_model:
        model.avgpool.register_forward_hook(get_activation('avgpool'))
    else:
        model.global_pool.register_forward_hook(get_activation('global_pool'))

    # get the features before the FC layer
    with torch.no_grad():
        for i, (sample, target) in enumerate(data_loader):
            sample = sample.to(device)
            sample_output = model(sample)
            if timm_model:
                sample_feature = activation['global_pool'].cpu()
            else:
                sample_feature = activation['avgpool'].cpu()
            
            #if not str_model:
            sample_feature = sample_feature.view(sample_feature.size(0), -1)
            #if madry_model:
            #    sample_feature = torch.flatten(sample_feature, 1)
            if data_features_tensor is None:
                data_features_tensor = sample_feature
                data_labels_tensor = target
            else:
                data_features_tensor = torch.cat((data_features_tensor, sample_feature))
                data_labels_tensor = torch.cat((data_labels_tensor, target))
            if i % 100==0:
                print("extracted for {} batches".format(i))
    #tensor_features_data = torch.utils.data.TensorDataset(data_features_tensor, data_labels_tensor)
    return data_features_tensor, data_labels_tensor


def extract_mobilenet_features(model, data_loader, device, amc=False, update_bn_stats=False, bn_updates=10):
    
    # this is where the extracted features will be added:
    data_features_tensor = None
    data_labels_tensor = None

    if update_bn_stats:
        # if true, update the BatchNorm stats by doing a few dummy forward passes through the data
        # if false, use the ImageNet Batch Norm stats
        model.train()
        for i in range(bn_updates):
            with torch.no_grad():
                for sample, target in data_loader:
                    sample = sample.to(device)
                    model(sample)
    model.eval()

    #register a forward hook:
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    if amc:
        model.features.register_forward_hook(get_activation('features'))
    else:
        model.model.register_forward_hook(get_activation('model'))

    # get the features before the FC layer
    with torch.no_grad():
        for i, (sample, target) in enumerate(data_loader):
            sample = sample.to(device)
            sample_output = model(sample)
            if amc:
                sample_feature = activation['features'].cpu()
                sample_feature = sample_feature.mean(3).mean(2)
            else:
                sample_feature = activation['model'].cpu()
                sample_feature = sample_feature.view(sample_feature.size(0), -1)

            if data_features_tensor is None:
                data_features_tensor = sample_feature
                data_labels_tensor = target
            else:
                data_features_tensor = torch.cat((data_features_tensor, sample_feature))
                data_labels_tensor = torch.cat((data_labels_tensor, target))
            if i % 100==0:
                print("extracted for {} batches".format(i))
    return data_features_tensor, data_labels_tensor



def classification_num_classes(dataset):
    return {'cifar10': 10,
            'cifar100': 100,
            'mnist': 10,
            'imagenet': 1000,
            'celeba': 40,
            'full_celeba': 40,
            'backdoorceleba': 40,
            'awa2': 50,
            }.get(dataset, None)


def _classification_get_input_shape(dataset):
    if dataset=='imagenet':
        return 1, 3, 224, 224
    elif dataset in ('cifar10', 'cifar100'):
        return 1, 3, 32, 32
    elif dataset == 'mnist':
        return 1, 1, 28, 28
    elif dataset=='celeba':
        return 1, 3, 224, 224
    elif dataset=='full_celeba':
        return 1, 3, 224, 224
    elif dataset=='label_specific_celeba':
        return 1, 3, 224, 224
    elif dataset=='backdoor_celeba':
        return 1, 3, 224, 224
    elif dataset=='awa2':
        return 1, 3, 224, 224
    elif dataset=='waterbirds':
        return 1, 3, 224, 224
    else:
        raise ValueError("dataset %s is not supported" % dataset)


def __dataset_factory(dataset):
    return globals()[f'{dataset}_get_datasets']

def interpolation_flag(interpolation):
    if interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    raise ValueError("interpolation must be one of 'bilinear', 'bicubic'")

def get_datasets(dataset, dataset_dir, **kwargs):
    datasets_fn = __dataset_factory(dataset)
    return datasets_fn(dataset_dir, **kwargs)


def mnist_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear'):
    # interpolation not used, here for consistent call.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset


def cifar10_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010) 
    
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation), 
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)


    test_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)
                                        ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar100_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)
    
    means = (0.5071, 0.4867, 0.4408)
    stds = (0.2675, 0.2565, 0.2761)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(), 
                                         transforms.Normalize(means, stds)])

    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear'):
    interpolation = interpolation_flag(interpolation)
    print("getting imagenet datasets")
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation = interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    return train_dataset, test_dataset


def celeba_get_datasets(data_dir, use_data_aug=True, label_indices=None,
        interpolation='bilinear', return_test=False, **kwargs):
    interpolation = interpolation_flag(interpolation)
    #normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                std=[0.5, 0.5, 0.5])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If we are only using some of the labels, remove all the ones we don't need.
    target_transform=None
    if label_indices:
        target_transform = lambda x: x[[label_indices]]

    if use_data_aug:
        # See https://github.com/princetonvisualai/DomainBiasMitigation/blob/c432e751632bce2c7467ef22a6ffb44402b88684/models/celeba_core.py#L53
        train_transform = transforms.Compose([
                                              transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    train_dataset = datasets.CelebA(root=data_dir, split='train',
            target_type='attr', transform=train_transform, target_transform = target_transform, download=True)

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.CelebA(root=data_dir, split='valid', target_type='attr', transform=test_transform, target_transform=target_transform)
    if not return_test:
        return train_dataset, val_dataset
    test_dataset = datasets.CelebA(root=data_dir, split='test', target_type='attr', transform=test_transform, target_transform=target_transform)
    return train_dataset, val_dataset, test_dataset


def full_celeba_get_datasets(data_dir, use_data_aug=True, label_indices=None,   interpolation='bilinear', return_test=False, **kwargs):
    interpolation = interpolation_flag(interpolation)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If we are only using some of the labels, remove all the ones we don't need.
    target_transform=None
    if label_indices:
        target_transform = lambda x: x[[label_indices]]

    if use_data_aug:
        # See https://github.com/princetonvisualai/DomainBiasMitigation/blob/c432e751632bce2c7467ef22a6ffb44402b88684/models/celeba_core.py#L53
        train_transform = transforms.Compose([
                                              transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    # assumes the full CelebA data is already downloaded and unzipped -- otherwise unzipping is not implemented, since the format is 7z, not python friendly
    train_dataset = FullCelebA(root=data_dir, split='train', target_type='attr', transform=train_transform, target_transform = target_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.CelebA(root=data_dir, split='valid', target_type='attr', transform=test_transform, target_transform=target_transform)
    if not return_test:
        return train_dataset, val_dataset
    test_dataset = datasets.CelebA(root=data_dir, split='test', target_type='attr', transform=test_transform, target_transform=target_transform)
    return train_dataset, val_dataset, test_dataset


def label_specific_celeba_get_datasets(data_dir, use_data_aug=True, label_indices=None,   interpolation='bilinear', return_test=False, label_id=None, label_value=None, **kwargs):
    if label_id == None or label_value == None:
        raise ValueError("label and value not specified")
    interpolation = interpolation_flag(interpolation)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If we are only using some of the labels, remove all the ones we don't need.
    target_transform=None
    if label_indices:
        target_transform = lambda x: x[[label_indices]]

    if use_data_aug:
        # See https://github.com/princetonvisualai/DomainBiasMitigation/blob/c432e751632bce2c7467ef22a6ffb44402b88684/models/celeba_core.py#L53
        train_transform = transforms.Compose([
                                              transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    # assumes the full CelebA data is already downloaded and unzipped -- otherwise unzipping is not implemented, since the format is 7z, not python friendly
    train_dataset = LabelSpecificCelebA(root=data_dir, split='train', target_type='attr', transform=train_transform, target_transform = target_transform, label_id=label_id, label_value=label_value)

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.CelebA(root=data_dir, split='valid', target_type='attr', transform=test_transform, target_transform=target_transform)
    if not return_test:
        return train_dataset, val_dataset
    test_dataset = datasets.CelebA(root=data_dir, split='test', target_type='attr', transform=test_transform, target_transform=target_transform)
    return train_dataset, val_dataset, test_dataset

def backdoorceleba_get_datasets(data_dir, use_data_aug=True, label_indices=None,  interpolation='bilinear',
                                backdoor_type_train=None, backdoor_type_test=None,
                                in_backdoor_folder=None,
                                out_backdoor_folder=None,
                                backdoor_label=None,
                                backdoor_fracs_train=None, backdoor_fracs_test=None, return_test=False):
    interpolation = interpolation_flag(interpolation)
    normalization_mean=[0.5, 0.5, 0.5]
    normalization_std=[0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=normalization_mean, std=normalization_std)

    # If we are only using some of the labels, remove all the ones we don't need.
    target_transform=None
    if label_indices:
        target_transform = lambda x: x[[label_indices]]

    if use_data_aug:
        # See https://github.com/princetonvisualai/DomainBiasMitigation/blob/c432e751632bce2c7467ef22a6ffb44402b88684/models/celeba_core.py#L53
        train_transform = transforms.Compose([
                                              transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    print("Train backdoor type in datasets is \t", backdoor_type_train)
    train_dataset = BackdoorCelebA(root=data_dir, split='train', target_type='attr', transform=train_transform, target_transform = target_transform, 
                                   download=True,
                                   backdoor_type=backdoor_type_train,
                                   backdoor_label=backdoor_label,
                                   in_backdoor_folder=in_backdoor_folder, 
                                   out_backdoor_folder=out_backdoor_folder, 
                                   backdoor_fracs=backdoor_fracs_train,
                                   backdoor_normalization_mean=normalization_mean, backdoor_normalization_std=normalization_std)

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    print("Test backdoor type in datasets is \t", backdoor_type_test)
    

    val_dataset = BackdoorCelebA(root=data_dir, split='valid',
                                  target_type='attr', transform=test_transform,
                                  target_transform=target_transform, download=True,
                                  backdoor_type=backdoor_type_test,
                                  backdoor_label=backdoor_label,
                                  in_backdoor_folder=in_backdoor_folder, 
                                  out_backdoor_folder=out_backdoor_folder, 
                                  backdoor_fracs=backdoor_fracs_test,
                                  backdoor_normalization_mean=normalization_mean,
                                  backdoor_normalization_std=normalization_std)
    if not return_test:
        return train_dataset, val_dataset
    test_dataset = BackdoorCelebA(root=data_dir, split='test',
                                  target_type='attr', transform=test_transform, target_transform=target_transform, download=True,
                                  backdoor_type=backdoor_type_test,
                                  backdoor_label=backdoor_label,
                                  in_backdoor_folder=in_backdoor_folder, 
                                  out_backdoor_folder=out_backdoor_folder, 
                                  backdoor_fracs=backdoor_fracs_test,
                                  backdoor_normalization_mean=normalization_mean,
                                  backdoor_normalization_std=normalization_std)
    return train_dataset, val_dataset, test_dataset


def awa2_get_datasets(data_dir, use_data_aug=True, split_size=0.8, split_seed=0, interpolation='bilinear', label_indices=None, return_test=False):
    interpolation = interpolation_flag(interpolation)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])
    

    predicates_file = os.path.join(data_dir, "predicate-matrix-binary.txt")
    predicates_mtx = np.loadtxt(predicates_file)
    classes = pd.read_csv("/home/Datasets/Animals_with_Attributes2/classes.txt", sep="\t", header=None)
    classes.columns = ["id", "klass"]
    ccs = classes.reset_index().set_index("klass")
    train_image_dir = os.path.join(data_dir, "train_images") 
    train_classes = os.listdir(train_image_dir)
    train_classes.sort()
    train_predicates_mtx = predicates_mtx[ccs.loc[train_classes]["index"].values]
    ood_image_dir = os.path.join(data_dir, "ood_images")
    ood_classes = os.listdir(ood_image_dir)
    ood_classes.sort()
    ood_predicates_mtx = predicates_mtx[ccs.loc[ood_classes]["index"].values]
    if label_indices is None:
        train_target_transform = lambda x: train_predicates_mtx[x] 
        ood_target_transform = lambda x: ood_predicates_mtx[x] 
    else:
        train_target_transform = lambda x: train_predicates_mtx[x][[label_indices]] 
        ood_target_transform = lambda x: ood_predicates_mtx[x][[label_indices]] 

    train_image_dir = os.path.join(data_dir, "train_images") 
    full_dataset = datasets.ImageFolder(train_image_dir, train_transform, target_transform=train_target_transform)

    trn_idxs = np.loadtxt(os.path.join(data_dir, "train_indices.txt")).astype(int)
    train_dataset = torch.utils.data.Subset(full_dataset, trn_idxs)

    full_dataset_no_da = datasets.ImageFolder(train_image_dir, test_transform, target_transform=train_target_transform)
    val_idxs = np.loadtxt(os.path.join(data_dir, "test_indices.txt")).astype(int)
    val_dataset = torch.utils.data.Subset(full_dataset_no_da, tst_idxs)
    test_image_dir = os.path.join(data_dir, "ood_images")
    test_dataset = datasets.ImageFolder(test_image_dir, test_transform, target_transform=ood_target_transform)
    if not return_test:
        return train_dataset, val_dataset
    return train_dataset, val_dataset, test_dataset



