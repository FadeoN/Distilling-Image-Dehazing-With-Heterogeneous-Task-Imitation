
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import cyclical_lr

import torch.nn.functional as F
from vgg import VGGNetFeats

from resnet_models import ResBlock, ResidualInResiduals

class Teacher(nn.Module):
    """
    Takes hazy free ground truth image and learns to reconstruct it in an unsupervised way
    """
    def __init__(self, input_channels=3, inner_channels=64, block_count = 6):
        super(Teacher, self).__init__()

        output_channels = input_channels


        self.downsample = self._make_downsample_layer(input_channels, inner_channels)

        self.res_blocks = nn.ModuleList([ResBlock(inner_channels, inner_channels) for i in range(block_count)])

        self.upsample = self._make_upsample_layer(inner_channels, inner_channels)

        self.reconstruction = self._make_reconstruction_layer(inner_channels, output_channels)


    def forward(self, gt_image):


        rec = self.downsample(gt_image)

        for i, _ in enumerate(self.res_blocks):
            rec = self.res_blocks[i](rec)
        
        rec = self.upsample(rec)

        rec = self.reconstruction(rec)

        return rec

        
    def _make_reconstruction_layer(self, inlayer, outlayer, stride=1):

        return nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.Tanh(),
        )

    def _make_downsample_layer(self, inlayer, outlayer, stride=2):

        return nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(outlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )

    def _make_upsample_layer(self, inlayer, outlayer, stride=1):

        return nn.Sequential(
            nn.Conv2d(inlayer, inlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )


        
class Student(nn.Module):
    """
    Takes hazy image as input and outputs hazy free image
    """

    def __init__(self, input_channels=3, inner_channels=64, block_count=6):
        super(Student, self).__init__()

        output_channels = input_channels

        self.downsample = self._make_downsample_layer(input_channels, inner_channels)

        self.res_blocks = nn.ModuleList([ResidualInResiduals(inner_channels, block_count=3) for i in range(block_count)])

        self.upsample = self._make_upsample_layer(inner_channels, inner_channels)
        
        self.reconstruction = self._make_reconstruction_layer(inner_channels, output_channels)


    def forward(self, hazy):

        rec = self.downsample(hazy)

        for i, _ in enumerate(self.res_blocks):
            rec = self.res_blocks[i](rec)
        
        rec = self.upsample(rec)

        rec = self.reconstruction(rec)

        return rec



    def _make_reconstruction_layer(self, inlayer, outlayer, stride=1):

        return nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.Tanh(),
        )

    def _make_downsample_layer(self, inlayer, outlayer, stride=2):

        return nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(outlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )

    def _make_upsample_layer(self, inlayer, outlayer, stride=1):

        return nn.Sequential(
            nn.Conv2d(inlayer, inlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

class TeacherTrainer(nn.Module):

    def __init__(self, args):
        super(TeacherTrainer, self).__init__()

        self.teacher = Teacher()

        self.teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=args.teacher_lr)

        # cyclical learning rate
        # self.clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
        # self.scheduler = optim.lr_scheduler.lambdaLr(self.optim, [clr])
        self.teacher_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.teacher_optimizer)


        self.teacher_l1loss = nn.L1Loss()
    
    def backward(self, gt):

        rec = self.teacher.forward(gt)

        losses = dict()

        teacher_loss = self.teacher_l1loss(rec, gt)
        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        self.teacher_optimizer.step()

        losses["teacher_rec_loss"] = teacher_loss


        self.teacher_scheduler.step(teacher_loss)

        return losses

    
    def forward_loss(self, gt):

        rec = self.teacher.forward(gt)

        losses = {}

        teacher_reconstruction_loss = self.teacher_l1loss(rec, gt)

        losses["teacher_rec_loss"] = teacher_reconstruction_loss

        return losses

    def get_reconstructed_images(self, gt):
        """
        Get hazy free reconstructed images
        """
        
        rec = self.teacher.forward(gt)

        return rec


class Distilled(nn.Module):

    def __init__(self, args):
        super(Distilled, self).__init__()

        self.teacher = Teacher()

        self.student = Student()

        self.vgg19 = VGGNetFeats()


        self.student_optimizer = optim.Adam(self.student.parameters(), lr=args.student_lr)

        self.teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=args.teacher_lr)

        # cyclical learning rate
        # self.clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
        # self.scheduler = optim.lr_scheduler.lambdaLr(self.optim, [clr])
        self.teacher_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.teacher_optimizer)

        self.student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.student_optimizer)


        # Reconstruction Losses
        self.teacher_l1loss = nn.L1Loss()
        self.student_l1loss = nn.L1Loss()

        # Perceptual loss
        self.perceptual_loss = nn.L1Loss()

        # Representation Mimicking Loss
        self.mimicking_loss = nn.L1Loss()


    def forward(self, gt, hazy):
        

        rec_gt = self.teacher.forward(gt)

        rec_hazy_free = self.student.forward(hazy)


        results = {"rec_gt":rec_gt,
                    "rec_hazy_free":rec_hazy_free}

        return results

    def backward(self, gt, hazy, args):

        results_forward = self.forward(gt, hazy)
        rec_gt, rec_hazy_free = results_forward["rec_gt"], results_forward["rec_hazy_free"]


        losses = dict()

        teacher_recons_loss = self.teacher_l1loss(rec_gt, gt)

        self.teacher_optimizer.zero_grad()
        teacher_recons_loss.backward()
        self.teacher_optimizer.step()

        
        student_recons_loss = self.student_l1loss(rec_hazy_free, gt)

        gt_perceptual_features = self.vgg19(gt)
        reconstructed_perceptual_features = self.vgg19(rec_hazy_free)

        perceptual_loss = 0.0

        # Sum up perceptual loss taken from different layers of VGG19 
        for idx, (gt_feat, rec_feat) in enumerate(zip(gt_perceptual_features, reconstructed_perceptual_features)):
            
            perceptual_loss += self.perceptual_loss(rec_feat, gt_feat)


        # TODO ADD MIMICKING LOSS

        self.student_optimizer.zero_grad()
        dehazing_loss = student_recons_loss + args.lambda_p * perceptual_loss
        dehazing_loss.backward()
        self.student_optimizer.step()
        



        losses["teacher_rec_loss"] = teacher_recons_loss
        losses["student_rec_loss"] = student_recons_loss

        losses["perceptual_loss"] = perceptual_loss
        losses["dehazing_loss"] = dehazing_loss


        self.teacher_scheduler.step(teacher_recons_loss)
        self.student_scheduler.step(student_recons_loss)

        return losses



    def forward_loss(self, gt, hazy, args):

        results_forward = self.forward(gt, hazy)    
        rec_gt, rec_hazy_free = results_forward["rec_gt"], results_forward["rec_hazy_free"]


        losses = dict()

        teacher_recons_loss = self.teacher_l1loss(rec_gt, gt)
        
        student_recons_loss = self.student_l1loss(rec_hazy_free, gt)

        gt_perceptual_features = self.vgg19(gt)
        reconstructed_perceptual_features = self.vgg19(rec_hazy_free)

        perceptual_loss = 0.0

        # Sum up perceptual loss taken from different layers of VGG19 
        for idx, (gt_feat, rec_feat) in enumerate(zip(gt_perceptual_features, reconstructed_perceptual_features)):
            
            perceptual_loss += self.perceptual_loss(rec_feat, gt_feat)


        # TODO ADD MIMICKING LOSS
        dehazing_loss = student_recons_loss + args.lambda_p * perceptual_loss

        



        losses["teacher_rec_loss"] = teacher_recons_loss
        losses["student_rec_loss"] = student_recons_loss

        losses["perceptual_loss"] = perceptual_loss
        losses["dehazing_loss"] = dehazing_loss


        self.teacher_scheduler.step(teacher_recons_loss)
        self.student_scheduler.step(student_recons_loss)

        return losses

    def get_reconstructed_images(self, gt, hazy):
        """
        Get hazy free reconstructed images
        """
        
        results_forward = self.forward(gt, hazy)    
        rec_gt, rec_hazy_free = results_forward["rec_gt"], results_forward["rec_hazy_free"]
        
        return rec_gt, rec_hazy_free