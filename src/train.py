import numpy as np
import time
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from options import Options
from models import Distilled
from data import DataGeneratorPaired
from logger import AverageMeter, Logger
from test import validate, save_image_results
import utils


np.random.seed(0)

def main():

    job_time =  time.strftime("%d_%m_%Y_%H_%M_%S")

    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    # Read the config file and
    config = utils.read_config()
    path_dataset = config['path_dataset']
    path_aux = config['path_aux']

    model_name = '+'.join(args.model_name)
    path_cp = os.path.join(path_aux, "CheckPoints", args.dataset, model_name, job_time)
    path_log = os.path.join(path_aux, 'LogFiles', args.dataset, model_name, job_time)
    path_results = os.path.join(path_aux, 'Results', args.dataset, model_name, job_time)

    os.makedirs(path_cp, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)
    os.makedirs(path_results, exist_ok=True)

    print('Checkpoint path: {}'.format(path_cp))
    print('Logger path: {}'.format(path_log))

    # Load the dataset
    print('Loading data...', end='')
    splits = utils.load_files_and_partition(path_dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio)


    train_data = DataGeneratorPaired(splits, mode="train")
    val_data = DataGeneratorPaired(splits, mode="val")
    test_data = DataGeneratorPaired(splits, mode="test")

    print("Done")


    # PyTorch train loader
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)

    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True)


    distilled_model = Distilled(args)

    # Load the whole model if exists, otherwise check whether teacher model exists
    if(args.load_best_model==True and args.best_model_path!="" and os.path.exists(args.best_model_path)):
        print("Loading Best model....")
        model_checkpoint = torch.load(args.best_model_path)
        distilled_model.load_state_dict(model_checkpoint)
        print("Done")
    elif(args.load_best_teacher_model == True and args.best_teacher_model_path!= "" and os.path.exists(args.best_teacher_model_path)):
        print("Loading Best teacher model....")
        model_checkpoint = torch.load(args.best_teacher_model_path)
        distilled_model.load_state_dict(model_checkpoint, strict=False)
        print("Done")


    print('Setting logger...', end='')
    logger = Logger(path_log, force=True)
    print('Done')


    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if args.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        distilled_model = distilled_model.cuda()
    print('Done')


    best_dehazing_loss = 0
    early_stop_counter = 0

    # Epoch for loop
    if not args.test:
        print('***Train***')
        for epoch in range(args.epochs):


            # train on training set
            losses = train(train_loader, distilled_model, epoch, args)

            # evaluate on validation set, map_ since map is already there
            print('***Validation***')
            valid_loss = validate(val_loader, distilled_model, epoch, args)
            dehazing_loss = valid_loss['dehazing_loss'].avg

            print('Dehazing Loss on validation set after {0} epochs: {1:.4f} (dehaze)'
                .format(epoch + 1, dehazing_loss))


            if dehazing_loss > best_dehazing_loss:
                best_dehazing_loss = dehazing_loss
                early_stop_counter = 0
                utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': distilled_model.state_dict(), 'best_dehazing_loss':
                    best_dehazing_loss}, directory=path_cp)
            else:
                if args.early_stop == early_stop_counter:
                    break
                early_stop_counter += 1

            # Logger step

            logger.add_scalar('teacher training reconstruction loss', losses['teacher_rec_loss'].avg)
            logger.add_scalar('teacher validation reconstruction loss', valid_loss['teacher_rec_loss'].avg)
            
            logger.add_scalar('student training reconstruction loss', losses['student_rec_loss'].avg)
            logger.add_scalar('student validation reconstruction loss', valid_loss['student_rec_loss'].avg)

            logger.add_scalar('student training perceptual loss', losses['perceptual_loss'].avg)
            logger.add_scalar('student validation perceptual loss', valid_loss['perceptual_loss'].avg)

            logger.add_scalar('dehazing training  loss', losses['dehazing_loss'].avg)
            logger.add_scalar('dehazing validation  loss', valid_loss['dehazing_loss'].avg)
            logger.step()


    # load the best model yet
    best_model_file = os.path.join(path_cp, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_dehazing_loss = checkpoint['best_dehazing_loss']
        distilled_model.load_state_dict(checkpoint['state_dict'])
        print("Loaded best model '{0}' (epoch {1}; Dehaze Network Loss {2:.4f})".format(best_model_file, epoch, best_dehazing_loss))
        print('***Test***')
        valid_loss = validate(test_loader, distilled_model, epoch, args)

        print('Results on test set: Dehaze Network Loss = {0:.4f}'.format(valid_loss['dehazing_loss'].avg))
        if args.save_image_results:
            print('Saving image results...', end='')
            path_image_results = os.path.join(path_results, 'image_results')
            save_image_results(test_loader, distilled_model, path_image_results)
            print('Done')
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


    




def train(train_loader, distilled_model, epoch, args):

    distilled_model.train()

    batch_time = AverageMeter()
    loss_teacher_rec = AverageMeter()
    loss_student_rec = AverageMeter()
    loss_student_perceptual = AverageMeter()
    loss_dehazing_network = AverageMeter()


    # Start counting time
    time_start = time.time()

    for i, item in enumerate(tqdm(train_loader)):

        gt, hazy = item["gt"], item["hazy"]

        if torch.cuda.is_available():
            gt, hazy = gt.cuda(), hazy.cuda()

        loss = distilled_model.backward(gt, hazy, args)

        loss_teacher_rec.update(loss["teacher_rec_loss"].item(), gt.size(0))
        loss_student_rec.update(loss["student_rec_loss"].item(), gt.size(0))
        loss_student_perceptual.update(loss["perceptual_loss"].item(), gt.size(0))
        loss_dehazing_network.update(loss["dehazing_loss"].item(), gt.size(0))

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % args.log_interval == 0:
            print('[Train] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Teacher Reconstruction Loss {loss_teacher.val:.4f} ({loss_teacher.avg:.4f})\t'
                  'Student Reconstruction Loss {loss_student.val:.4f} ({loss_student.avg:.4f})\t'
                  'Student Perceptual Loss {loss_perc.val:.4f} ({loss_perc.avg:.4f})\t'
                    'Dehazing Network Loss {loss_dehaze.val:.4f} ({loss_dehaze.avg:.4f})\t'
                  .format(epoch + 1, i + 1, len(train_loader), batch_time=batch_time, loss_teacher=loss_teacher_rec,
                  loss_student=loss_student_rec, loss_perc=loss_student_perceptual, loss_dehaze=loss_dehazing_network))



    losses = {"teacher_rec_loss":loss_teacher_rec,
                "student_rec_loss":loss_student_rec,
                "perceptual_loss":loss_student_perceptual,
                "dehazing_loss":loss_dehazing_network}

    return losses



if __name__ == '__main__':
    main()