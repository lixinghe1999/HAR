from models.mn.model import get_model as get_mobilenet_model
from models.mn.model import NAME_TO_WIDTH
from models.mn.preprocess import AugmentMelSTFT

import torch.nn as nn

class Mobilenet_Encoder(nn.Module):
    def __init__(self, model_name = 'mn10_as'):
        super().__init__()
        self.model = get_mobilenet_model(pretrained_name=model_name, width_mult=NAME_TO_WIDTH(model_name), 
                                         strides=[2, 2, 2, 2], head_type='mlp')
        self.preprocess = AugmentMelSTFT()

    def forward(self, x, return_fmaps=False):
        x = self.preprocess(x)
        x, feature = self.model(x.unsqueeze(1), return_fmaps=return_fmaps)
        return x, feature