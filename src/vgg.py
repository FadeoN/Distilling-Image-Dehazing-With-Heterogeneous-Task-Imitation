import torch.nn as nn
from torchvision import models

class VGGNetFeats(nn.Module):
    def __init__(self, pretrained=True, finetune=False, selected_layers=[2, 7, 12, 21, 30]):
        """
        selected_layers => choose which layers to use to extract features
        """
        super(VGGNetFeats, self).__init__()

        self.selected_layers = selected_layers

        model = models.vgg19(pretrained=pretrained)

        for param in model.parameters():
            param.requires_grad = finetune


        self.features = model.features



    def forward(self, x):

        outs = []

        for idx, layer in enumerate(self.features):
            
            x = layer(x)

            if idx in self.selected_layers:
                outs.append(x)

        return tuple(outs)



