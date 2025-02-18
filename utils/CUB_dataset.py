import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset

def get_CUB_train_test(train_bs, test_bs, NW, SEG=False):
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

    data_dir = '../datasets/CUB/CUB_200_2011/'
    train_test_split = []
    with open(data_dir + 'train_test_split.txt', 'r') as file:
        for l in file.readlines():
            train_test_split.append(int(l[:-1].split(' ')[1]))
    train_idx = [index for index, value in enumerate(train_test_split) if value == 1]
    test_idx = [index for index, value in enumerate(train_test_split) if value == 0]

    # Training data
    cub_train_dataset = ImageFolder(root=data_dir + 'images/', transform=train_transform)
    cub_train_subset_dataset = Subset(cub_train_dataset, train_idx)
    cub_train_dataloader = DataLoader(cub_train_subset_dataset, batch_size=train_bs,
                                      shuffle=True, num_workers=NW, pin_memory=True, drop_last=True)
    # Testing data
    cub_test_dataset = ImageFolder(root=data_dir + 'images/', transform=test_transform)
    cub_test_subset_dataset = Subset(cub_test_dataset, test_idx)
    cub_test_dataloader = DataLoader(cub_test_subset_dataset, batch_size=test_bs,
                                     shuffle=False, num_workers=NW, pin_memory=True, drop_last=False)
    if SEG:
        seg_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        seg_dir = '../datasets/CUB/segmentations/'
        cub_seg_test_dataset = ImageFolder(root=seg_dir, transform=seg_transform)
        cub_seg_test_subset_dataset = Subset(cub_seg_test_dataset, test_idx)
        cub_seg_test_dataloader = DataLoader(cub_seg_test_subset_dataset, batch_size=test_bs,
                                             shuffle=False, num_workers=NW, pin_memory=True, drop_last=False)
        return cub_test_dataloader, cub_seg_test_dataloader, len(cub_test_subset_dataset)
    else:
        return cub_train_dataloader, cub_test_dataloader, len(cub_test_subset_dataset)