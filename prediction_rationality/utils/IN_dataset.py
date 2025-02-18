import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import os
from torch.utils.data.distributed import DistributedSampler

def get_IN_train_val(CN, train_bs, val_bs, NW, mg=False):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    data_dir_train = '../datasets/IN/ilsvrc/train/'
    data_dir_val = '../datasets/IN/ilsvrc/val/'
    imagenet_dataset_train = ImageFolder(root=data_dir_train, transform=train_transform)
    imagenet_dataset_val = ImageFolder(root=data_dir_val, transform=test_transform)

    if mg:
        train_sampler = DistributedSampler(imagenet_dataset_train, shuffle=True, drop_last=True)
        imagenet_train_dataloader = torch.utils.data.DataLoader(
            imagenet_dataset_train, sampler=train_sampler, batch_size=train_bs, num_workers=NW, pin_memory=True
        )
        val_sampler = DistributedSampler(imagenet_dataset_val, shuffle=False, drop_last=False)
        imagenet_val_dataloader = torch.utils.data.DataLoader(
            imagenet_dataset_val, batch_size=val_bs, sampler=val_sampler, num_workers=NW, pin_memory=True
        )
    else:
        imagenet_train_dataloader = DataLoader(imagenet_dataset_train,
                                               batch_size=train_bs, shuffle=True, drop_last=True,
                                               num_workers=NW, pin_memory=True)
        imagenet_val_dataloader = DataLoader(imagenet_dataset_val,
                                             batch_size=val_bs, shuffle=False, num_workers=NW,
                                             pin_memory=True, drop_last=False)

    image_paths = [str(root) for root, filename in imagenet_dataset_val.imgs]
    # image_paths_f = [ip_str.rsplit('/', 1)[0] for ip_str in image_paths]

    return imagenet_train_dataloader, imagenet_val_dataloader, len(imagenet_dataset_val), image_paths

def get_INc_train_val(CN, val_bs, variant_name, M, NW):
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    all_classes = list(range(1000))

    data_dir = '../datasets/IN/ImageNet-C/' + variant_name + '/' + str(M) + '/'
    imagenet_dataset = ImageFolder(root=data_dir, transform=test_transform)
    # for bbox size only:
    data_dir_ori = '../datasets/IN/ilsvrc/val/'
    imagenet_dataset_ori = ImageFolder(root=data_dir_ori, transform=test_transform)
    image_paths = [str(root) for root, filename in imagenet_dataset_ori.imgs]
    # image_paths_f = [ip_str.rsplit('/', 1)[0] for ip_str in image_paths]
    batch_size = val_bs
    imagenet_val_dataloader = DataLoader(imagenet_dataset,
                                         batch_size=batch_size, shuffle=False, num_workers=NW,
                                         pin_memory=True, drop_last=False)
    return imagenet_val_dataloader, len(imagenet_dataset), image_paths