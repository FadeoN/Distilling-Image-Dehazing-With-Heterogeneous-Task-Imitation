import numpy as np
import os 
import glob
import configparser as cp
import socket
import math
import torch
import shutil
from torchvision.utils import save_image

# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

def read_config():
    config = cp.ConfigParser()
    cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config.read(os.path.join(cur_path, 'config.ini'))
    host = socket.gethostname()
    return config[host]

def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x

def save_checkpoint(state, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)

def load_files_and_partition(root_path, train_ratio=0.9, val_ratio=0.05):
    
    np.random.seed(0)

    gt_paths, hazy_paths = load_pairs(root_path)



    if len(gt_paths) != len(hazy_paths):
        raise "Inconsistent dataset length"

    indexes = list(range(len(gt_paths)))

    tr_indexes = np.random.choice(indexes, int(train_ratio * len(indexes)), replace=False)
    val_indexes = np.random.choice(np.setdiff1d(indexes, tr_indexes), int(val_ratio * len(indexes)), replace=False)
    te_indexes = np.setdiff1d(indexes, np.union1d(tr_indexes, val_indexes))


    splits = dict()

    splits["tr_gt_paths"] = gt_paths[tr_indexes]
    splits["val_gt_paths"] = gt_paths[val_indexes]
    splits["test_gt_paths"] = gt_paths[te_indexes]


    splits["tr_hazy_paths"] = hazy_paths[tr_indexes]
    splits["val_hazy_paths"] = hazy_paths[val_indexes]
    splits["test_hazy_paths"] = hazy_paths[te_indexes]

    return splits

def load_pairs(root_path):

    return (load_gt_images(root_path), load_hazy_images(root_path))

def load_gt_images(root_path):

    return np.array([name for name in glob.glob(os.path.join(root_path, "*/*_GT.jpg"))])

    
def load_hazy_images(root_path):
    return np.array([name for name in glob.glob(os.path.join(root_path, "*/*_hazy.jpg"))])

def save_an_image(path, path_results, img, postfix="_REC"):

    base, ext = os.path.basename(path).split(".")

    base = base + postfix

    img_name = base + "." + ext
    path = os.path.join(path_results, img_name)



    save_image(img, path, normalize=True, range=(-1, 1))