import os
import numpy as np
import cv2
from torchvision import datasets
from scipy.io import loadmat
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json

Img2AnnoDict = {
    "airplanes": "Airplanes_Side_2",
    "Faces": "Faces_2",
    "Faces_easy": "Faces_3",
    "Motorbikes": "Motorbikes_16"
}

class CustomImageMatDataset(Dataset):
    def __init__(self, root_dir, transform=None, seg_mask_transform=None):
        self.image_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.root_dir = root_dir
        self.transform = transform
        self.seg_mask_transform = seg_mask_transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img_path, _ = self.image_dataset.imgs[idx]
        label = self.image_dataset.targets[idx]
        mat_path = img_path.replace("101_ObjectCategories/", "Annotations/").replace("image_", "annotation_")

        image = Image.open(img_path)
        # img_ori = image
        mat_file_path = os.path.splitext(mat_path)[0] + '.mat'
        class_name = mat_file_path.split('/')[-2]
        if class_name in Img2AnnoDict.keys():
            mat_file_path = mat_file_path.replace(class_name, Img2AnnoDict[class_name])
            # print(mat_file_path)
        try:
            mat_data = loadmat(mat_file_path)
            img_w, img_h = image.size

            seg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # print(mat_data['box_coord'].shape, mat_data['box_coord'].shape)
            h, w = mat_data['box_coord'][0][2], mat_data['box_coord'][0][0]
            res_cont = mat_data['obj_contour'].astype(np.int32).T + np.array([h, w]).astype(np.int32)
            cv2.fillPoly(seg_mask, [res_cont], 1)
            # seg_mask = cv2.resize(seg_mask, (224, 224))
        except FileNotFoundError:
            seg_mask = None

        if self.transform:
            try:
                image = self.transform(image)
            except:
                image = Image.fromarray(cv2.cvtColor(np.array(image)[:, :, np.newaxis], cv2.COLOR_GRAY2RGB))
                image = self.transform(image)
            if seg_mask is not None and self.seg_mask_transform is not None:
                seg_mask_pil = Image.fromarray(seg_mask)
                seg_mask_pil = self.seg_mask_transform(seg_mask_pil)
                seg_mask = np.array(seg_mask_pil)

        return image, label, img_path, seg_mask

def get_CT101_train_val(train_bs, test_bs, NW, test_ratio):

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

    img_dir = '../datasets/caltech_101/caltech-101/101_ObjectCategories/'
    caltech101_dataset_1 = CustomImageMatDataset(
        img_dir, transform=train_transform, seg_mask_transform=seg_mask_transform
    )
    caltech101_dataset_2 = CustomImageMatDataset(
        img_dir, transform=test_transform, seg_mask_transform=seg_mask_transform
    )
    with open('../datasets/caltech_101/yes_seg.json', 'r') as file:
        yes_seg = json.load(file)
    sample_size = int(len(yes_seg) * test_ratio)
    step = len(yes_seg) // sample_size
    sampled_indices = []
    for i in range(0, len(yes_seg), step):
        sampled_indices.append(i + 1)
    test_idxs = [yes_seg[i] for i in sampled_indices]
    train_idxs = list(set(yes_seg) - set(test_idxs))

    caltech101_dataset_train = Subset(caltech101_dataset_1, train_idxs)
    caltech101_dataset_test = Subset(caltech101_dataset_2, test_idxs)

    train_dataloader = DataLoader(caltech101_dataset_train, batch_size=train_bs, shuffle=True,
                                  num_workers=NW, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(caltech101_dataset_test, batch_size=test_bs, shuffle=False,
                                 num_workers=NW, pin_memory=True, drop_last=False)

    return train_dataloader, test_dataloader, len(caltech101_dataset_test)

def filter_CT101():
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
    img_dir = '../datasets/caltech_101/caltech-101/101_ObjectCategories/'
    caltech101_dataset = CustomImageMatDataset(
        img_dir, transform=test_transform, seg_mask_transform=seg_mask_transform
    )
    yes_seg, no_seg = [], []
    idx = 0
    label_dict = {}
    for ele in caltech101_dataset:
        image, label, img_path, seg_mask = ele
        if label not in label_dict:
            label_dict[label] = img_path.split('/')[-2].replace("_", " ").lower()
        if seg_mask is not None:
            yes_seg.append(idx)
        else:
            no_seg.append(idx)
        idx += 1
    return yes_seg, no_seg, label_dict