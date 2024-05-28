import os
from typing import Callable

import numpy as np
import torch
import torchvision.transforms as tf
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


def get_files(path, ext):
    return [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)
    ]


class TransformOCTBilinear(object):
    def __new__(cls, img_size=(128, 128), *args, **kwargs):
        return tf.Compose([tf.Resize(img_size)])


def get_full_dataset(data_path, img_size):
    """
    Loads the full dataset from train, val, and test directories and applies required transformations.

    Args:
    data_path (str): Base path to the dataset directory.
    img_size (int): Target size for image resizing.

    Returns:
    Dataset: A combined dataset containing all images.
    """
    size_transform = TransformOCTBilinear(img_size=(img_size, img_size))
    img_transform = None

    # Paths to different sets
    train_dataset_path = os.path.join(data_path, "train")
    val_dataset_path = os.path.join(data_path, "val")
    test_dataset_path = os.path.join(data_path, "test")

    # Load datasets
    train_dataset = DatasetOct(
        train_dataset_path,
        size_transform=size_transform,
        normalized=True,
        image_transform=img_transform,
    )
    val_dataset = DatasetOct(
        val_dataset_path,
        size_transform=size_transform,
        normalized=True,
        image_transform=img_transform,
    )
    test_dataset = DatasetOct(
        test_dataset_path,
        size_transform=size_transform,
        normalized=True,
        image_transform=img_transform,
    )

    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset(
        [train_dataset, val_dataset, test_dataset]
    )

    return combined_dataset


def get_k_fold_data(k, data_path, img_size, batch_size, seed):
    # Load the full dataset
    full_dataset = get_full_dataset(data_path, img_size)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_index, val_index in kf.split(full_dataset):
        # Create train and validation datasets using indices
        train_data = torch.utils.data.Subset(full_dataset, train_index)
        val_data = torch.utils.data.Subset(full_dataset, val_index)

        # Create DataLoaders for each fold
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=1, shuffle=False
        )  # Batch size 1 for validation

        yield train_loader, val_loader


def get_data(data_path, img_size, batch_size, val_batch_size=1):
    train_dataset_path = os.path.join(data_path, "train")
    val_dataset_path = os.path.join(data_path, "val")
    test_dataset_path = os.path.join(data_path, "test")

    size_transform = TransformOCTBilinear(img_size=(img_size, img_size))

    img_transform = None

    train_dataset = DatasetOct(
        train_dataset_path,
        size_transform=size_transform,
        normalized=True,
        image_transform=img_transform,
    )
    val_dataset = DatasetOct(
        val_dataset_path, size_transform=size_transform, normalized=True
    )
    test_dataset = DatasetOct(
        test_dataset_path, size_transform=size_transform, normalized=True
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return trainloader, valloader, testloader, train_dataset, val_dataset, test_dataset


class TransformStandardization(object):
    """
    Standardizaton / z-score: (x-mean)/std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + f": mean {self.mean}, std {self.std}"


class TransformOCTMaskAdjustment(object):
    """
    Adjust OCT 2015 Mask
    from: classes [0,1,2,3,4,5,6,7,8,9], where 9 is fluid, 0 is empty space above and 8 empty space below
    to: class 0: not class, classes 1-7: are retinal layers, class 8: fluid
    """

    def __call__(self, mask):
        mask[mask == 8] = 0
        mask[mask == 9] = 8
        return mask


class DatasetOct(Dataset):
    """
    Map style dataset object for oct 2015 data
    - expects .npy files
    - assumes sliced images, masks - produced by our project: dataloading/preprocessing.py
        (valid dims of images,masks and encoding: pixel label e [0..9], for every pixel)

    Parameters:
        dataset_path: path to the dataset path/{images,masks}
        size_transform: deterministic transformation for resizing applied to image and mask separately
        joint_transform: random transformations applied to image and mask jointly after size_transform
        image_transform: transformation applied only to the image and after joint_transform
    _getitem__(): returns image and corresponding mask
    """

    def __init__(
        self,
        dataset_path: str,
        joint_transform: Callable = None,
        size_transform: Callable = None,
        image_transform: Callable = None,
        normalized=True,
    ) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, "images")
        self.output_path = os.path.join(dataset_path, "masks")
        self.images_list = get_files(self.input_path, ".npy")

        # size transform
        self.size_transform = size_transform

        self.joint_transform = joint_transform

        self.mask_adjust = TransformOCTMaskAdjustment()

        self.image_transform = image_transform

        self.normalized = normalized
        # gray scale oct 2015: calculated with full tensor in memory {'mean': tensor([46.3758]), 'std': tensor([53.9434])}
        # calculated with batched method {'mean': tensor([46.3756]), 'std': tensor([53.9204])}
        self.normalize = TransformStandardization(
            (46.3758), (53.9434)
        )  # torchvision.transforms.Normalize((46.3758), (53.9434))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        img = np.load(os.path.join(self.input_path, image_filename))
        mask = np.load(os.path.join(self.output_path, image_filename))

        # img_size 128 works - general transforms require (N,C,H,W) dims
        img = img.squeeze()
        mask = mask.squeeze()

        img = torch.Tensor(img).reshape(1, 1, *img.shape)
        mask = torch.Tensor(mask).reshape(1, 1, *mask.shape).int()

        # adjust mask classes
        mask = self.mask_adjust(mask)

        # note for some reason some masks differ in size from the actual image (dims)
        if self.size_transform:
            img = self.size_transform(img)
            mask = self.size_transform(mask)

        # normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        if self.joint_transform:
            img, mask = self.joint_transform([img, mask])

        # img = img.reshape(1, img.shape[2], img.shape[3])
        if self.image_transform:
            img = self.image_transform(img)

        # img = img.reshape(1, *img.shape)

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze(dim=1).long()

        return img, mask
