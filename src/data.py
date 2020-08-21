
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import glob
from PIL import Image

class DataGeneratorPaired(data.Dataset):

    def __init__(self, splits, mode="train"):


        if mode == "train":
            self.gt_paths = splits["tr_gt_paths"]
            self.hazy_paths = splits["tr_hazy_paths"]
        elif mode == "val":
            self.gt_paths = splits["val_gt_paths"]
            self.hazy_paths = splits["val_hazy_paths"]
        elif mode == "test":
            self.gt_paths = splits["test_gt_paths"]
            self.hazy_paths = splits["test_hazy_paths"]
        else:
            raise "Incorrect dataset mode"

        self.transform_hazy = self.get_transforms()
        self.transform_gt = self.get_transforms()


    def get_transforms(self):
        """
        Returns transforms to apply on images.
        """

        transforms_list = []

        transforms_list.append(transforms.CenterCrop(255))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ))
        return transforms.Compose(transforms_list)


    def __getitem__(self, index):

        gt = Image.open(self.gt_paths[index])
        hazy = Image.open(self.hazy_paths[index])


        if self.transform_hazy is not None:
            hazy = self.transform_hazy(hazy)

        if self.transform_gt is not None:
            gt = self.transform_gt(gt)

        item = {"hazy":hazy,
                "gt":gt,
                "gt_paths":self.gt_paths[index],
                "hazy_paths":self.hazy_paths[index]}

        return item

    def __len__(self):

        return len(self.gt_paths)

    