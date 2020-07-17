import numpy as np
import time
import os
from tqdm import tqdm

from logger import AverageMeter, Logger
import utils

import torch
from torchvision import transforms
from PIL import Image
import utils

np.random.seed(0)

def validate(val_loader, distilled_model, epoch, args):

	# Switch to test mode
    distilled_model.eval()

    batch_time = AverageMeter()
    loss_teacher_rec = AverageMeter()
    loss_student_rec = AverageMeter()
    loss_student_perceptual = AverageMeter()
    loss_dehazing_network = AverageMeter()
    loss_psnr = AverageMeter()
    loss_ssim = AverageMeter()


    # Start counting time
    time_start = time.time()
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(val_loader)):


            gt, hazy = item["gt"], item["hazy"]

            if torch.cuda.is_available():
                gt, hazy = gt.cuda(), hazy.cuda()

            loss = distilled_model.forward_loss(gt, hazy, args)

            loss_teacher_rec.update(loss["teacher_rec_loss"].item(), gt.size(0))
            loss_student_rec.update(loss["student_rec_loss"].item(), gt.size(0))
            loss_student_perceptual.update(loss["perceptual_loss"].item(), gt.size(0))
            loss_dehazing_network.update(loss["dehazing_loss"].item(), gt.size(0))
            loss_psnr.update(loss["loss_psnr"].item(), gt.size(0))
            loss_ssim.update(loss["loss_ssim"].item(), gt.size(0))


            # time
            time_end = time.time()
            batch_time.update(time_end - time_start)
            time_start = time_end

            if (i + 1) % args.log_interval == 0:
                print('[Test] Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Teacher Reconstruction Loss {loss_teacher.val:.4f} ({loss_teacher.avg:.4f})\t'
                    'Student Reconstruction Loss {loss_student.val:.4f} ({loss_student.avg:.4f})\t'
                    'Student Perceptual Loss {loss_perc.val:.4f} ({loss_perc.avg:.4f})\t'
                    'Dehazing Network Loss {loss_dehaze.val:.4f} ({loss_dehaze.avg:.4f})\t'
                    'PSNR {loss_psnr.val:.4f} ({loss_psnr.avg:.4f})\t'
                    'SSIM {loss_ssim.val:.4f} ({loss_ssim.avg:.4f})\t'
                    .format(epoch + 1, i + 1, len(val_loader), batch_time=batch_time, loss_teacher=loss_teacher_rec,
                    loss_student=loss_student_rec, loss_perc=loss_student_perceptual, loss_dehaze=loss_dehazing_network,
                    loss_psnr=loss_psnr, loss_ssim=loss_ssim))



        losses = {"teacher_rec_loss":loss_teacher_rec,
                    "student_rec_loss":loss_student_rec,
                    "perceptual_loss":loss_student_perceptual,
                    "dehazing_loss":loss_dehazing_network,
                    "loss_psnr":loss_psnr,
                    "loss_ssim":loss_ssim}

        return losses


def save_image_results(test_loader, distilled_model, path_results):

    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    distilled_model.eval()

    with torch.no_grad():

        for i, item in enumerate(tqdm(test_loader)):

            gt = item["gt"]
            hazy = item["hazy"]
            
            gt_paths = item["gt_paths"]
            hazy_paths = item["hazy_paths"]

            if torch.cuda.is_available():
                gt, hazy = gt.cuda(), hazy.cuda()

            rec_gts, rec_frees = distilled_model.get_reconstructed_images(gt, hazy)


            for j, gt_path in enumerate(gt_paths):

                # Save Reconstructed GT image
                rec = rec_gts[j].cpu()
                utils.save_an_image(gt_path, path_results, rec, postfix="_REC")

            for j, free_path in enumerate(hazy_paths):

                # Save Reconstructed Hazy Free image
                rec = rec_frees[j].cpu()
                utils.save_an_image(free_path, path_results, rec, postfix="_FREE")
                



