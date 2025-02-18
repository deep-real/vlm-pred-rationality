import os
import numpy as np
import cv2
from torchvision import datasets
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import re
import json

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, seg_mask_transform=None):
        self.image_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.root_dir = root_dir
        self.transform = transform
        self.seg_mask_transform = seg_mask_transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img_path, _ = self.image_dataset.imgs[idx]
        image = Image.open(img_path)
        label = self.image_dataset.targets[idx]
        mat_file_path = os.path.splitext(img_path)[0].replace('Images', 'Annotation')
        x_min, x_max, y_min, y_max = -1, -1, -1, -1
        with open(mat_file_path, 'r') as f:
            for l in f.readlines():
                if "<xmin>" in l:
                    x_min = int(re.findall(r'\d+', l)[0])
                elif "<xmax>" in l:
                    x_max = int(re.findall(r'\d+', l)[0])
                elif "<ymin>" in l:
                    y_min = int(re.findall(r'\d+', l)[0])
                elif "<ymax>" in l:
                    y_max = int(re.findall(r'\d+', l)[0])
                else:
                    pass
        det_mask = np.zeros((np.array(image).shape[0], np.array(image).shape[1]), dtype=np.uint8)
        cv2.rectangle(det_mask, (x_min, y_min), (x_max, y_max), 1, thickness=-1)
        if np.array(image).shape[2] == 4:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
            det_mask = np.array(self.seg_mask_transform(Image.fromarray(det_mask)))
        return image, label, img_path, det_mask

def get_StanDog_train_val(train_bs, test_bs, NW):
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
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    seg_mask_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

    img_dir = '../datasets/Stanford_Dogs/Images/'
    standog_dataset_1 = CustomImageDataset(
        img_dir, transform=train_transform, seg_mask_transform=seg_mask_transform
    )
    standog_dataset_2 = CustomImageDataset(
        img_dir, transform=test_transform, seg_mask_transform=seg_mask_transform
    )
    with open('../datasets/Stanford_Dogs/train_idxs.json', 'r') as file:
        train_idxs = json.load(file)
    with open('../datasets/Stanford_Dogs/test_idxs.json', 'r') as file:
        test_idxs = json.load(file)
    standog_dataset_train = Subset(standog_dataset_1, train_idxs)
    standog_dataset_test = Subset(standog_dataset_2, test_idxs)

    train_dataloader = DataLoader(standog_dataset_train, batch_size=train_bs, shuffle=True,
                                  num_workers=NW, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(standog_dataset_test, batch_size=test_bs, shuffle=False,
                                 num_workers=NW, pin_memory=True, drop_last=False)

    return train_dataloader, test_dataloader, len(standog_dataset_test)

def get_standog_labels():
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    seg_mask_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])
    img_dir = '../datasets/Stanford_Dogs/Images/'
    standog_dataset = CustomImageDataset(
        img_dir, transform=test_transform, seg_mask_transform=seg_mask_transform
    )
    label_dict = {}
    for iter, ele in enumerate(standog_dataset):
        image, label, img_path, det_mask = ele
        if label not in label_dict:
            label_dict[label] = img_path.split('/')[-2].split('-')[1].replace("_", " ").lower()
    return label_dict