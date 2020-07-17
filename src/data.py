
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import glob
from PIL import Image

class DataGeneratorPaired(data.Dataset):

    def __init__(self, splits, mode="train", transform_hazy=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()]), 
    transform_gt=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()])):


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

        self.transform_hazy = transform_hazy
        self.transform_gt = transform_gt




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

    