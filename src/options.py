#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import utils
import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='Distilling Network for Image Dehazing')
        # Optional argument
        parser.add_argument('--dataset', required=False, default='O-Haze', help='Name of the dataset')

        # Semantic models
        parser.add_argument('--model-name', nargs='+', default=['teacher', 'student'],
                            type=str, help='Distilled Model')
        # Weight (on loss) parameters
        parser.add_argument('--lambda-p', default=1.0, type=float, help='Weight on the perceptual loss')
        parser.add_argument('--lambda-rm', default=1.0, type=float, help='Weight on the reconstruction loss')
        # model parameters
        parser.add_argument('--batch-size', default=4, type=int, help='Batch size')
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in data loader')
        parser.add_argument('--load-best-model', action='store_false', default=True, help='Load the model weight from given path.')
        parser.add_argument('--load-best-teacher-model', action='store_true', default=False, help='Load the teacher model weight from given path.')
        parser.add_argument("--best-model-path", type=str, default="model_best.pth", help="Best model path to be loaded.")
        parser.add_argument("--best-teacher-model-path", type=str, default="", help="Best teacher model path to be loaded.")

        # Checkpoint parameters
        parser.add_argument('--test', action='store_false', default=True, help='Test only flag')
        parser.add_argument('--early-stop', type=int, default=20, help='Early stopping epochs.')
        # Optimization parameters
        parser.add_argument('--epochs', type=int, default=30, metavar='N',
                            help='Number of epochs to train (default: 100)')
        parser.add_argument('--teacher-lr', type=lambda x: utils.restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='TLR',
                            help='Teacher Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
        parser.add_argument('--student-lr', type=lambda x: utils.restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='SLR',
                            help='Student Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
        parser.add_argument('--train-ratio', type=lambda x: utils.restricted_float(x, [0., 1.]), default=0.80, metavar='tr_ratio',
                            help='Training data ratio [0, 1] (default: 0.80)')
        parser.add_argument('--val-ratio', type=lambda x: utils.restricted_float(x, [0., 1.]), default=0.05, metavar='val_ratio',
                            help='Validation data ratio [0, 1] (default: 0.10)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        # I/O parameters
        parser.add_argument('--gt-dir', type=str, default="GT", 
                            help='Ground truth directory name of the images.')
        parser.add_argument('--hazy-dir', type=str, default="hazy",
                            help='Hazy directory name of the images.')

        parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                            help='How many batches to wait before logging training status')
        parser.add_argument('--save-image-results', action='store_false', default=True, help='Whether to save the image '
                                                                                            'results')                                                                                           
        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
