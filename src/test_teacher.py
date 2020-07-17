import numpy as np
import time
import os
from tqdm import tqdm

from logger import AverageMeter, Logger
import utils

import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image


def validate(val_loader, teacher_model, epoch, args):

	# Switch to test mode
	teacher_model.eval()

	batch_time = AverageMeter()
	loss_teacher_rec = AverageMeter()

	# Start counting time
	time_start = time.time()

	with torch.no_grad():

		for i, item in enumerate(tqdm(val_loader)):
			
			gt = item["gt"]

			if torch.cuda.is_available():
				gt = gt.cuda()

			# Sketch embedding into a semantic space
			loss = teacher_model.forward_loss(gt)

			loss_teacher_rec.update(loss["teacher_rec_loss"].item(), gt.size(0))

			# time
			time_end = time.time()
			batch_time.update(time_end - time_start)
			time_start = time_end

			if (i + 1) % args.log_interval == 0:
				print('[Validation] Epoch: [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Teacher Reconstruction Loss {loss_teacher.val:.4f} ({loss_teacher.avg:.4f})\t'
					.format(epoch + 1, i + 1, len(val_loader), batch_time=batch_time, loss_teacher=loss_teacher_rec))

	losses = {"teacher_rec_loss":loss_teacher_rec}

	return losses


def save_image_results(test_loader, teacher_model, path_results):

	if not os.path.isdir(path_results):
		os.makedirs(path_results)

	teacher_model.eval()

	with torch.no_grad():

		for i, item in enumerate(tqdm(test_loader)):

			gt = item["gt"]
			gt_paths = item["gt_paths"]


			if torch.cuda.is_available():
				gt = gt.cuda()

			reconstructed = teacher_model.get_reconstructed_images(gt)

			for j, gt_path in enumerate(gt_paths):

				img = reconstructed[j].cpu()

				utils.save_an_image(gt_path, path_results, img, postfix="_T_REC")