from models.mn.model import get_model as get_mobilenet_model
from models.mn.model import NAME_TO_WIDTH
from models.mn.preprocess import AugmentMelSTFT

import torch.nn as nn
import torch
import numpy as np

def labels():
    import csv

    # Load label
    with open('EfficientAT/metadata/class_labels_indices.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)

    labels = []
    ids = []    # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)

    classes_num = len(labels)
    return labels

class Mobilenet_Encoder(nn.Module):
    def __init__(self, model_name = 'mn10_as'):
        super().__init__()
        self.model = get_mobilenet_model(pretrained_name=model_name, width_mult=NAME_TO_WIDTH(model_name), 
                                         strides=[2, 2, 2, 2], head_type='mlp')
        self.preprocess = AugmentMelSTFT()
        self.labels = labels()
    def forward(self, x, return_fmaps=False):
        x = self.preprocess(x)
        x, feature = self.model(x.unsqueeze(1), return_fmaps=return_fmaps)
        return x, feature
    
    def default_tagging(self, x, window_size=16000*10):
        """
            Tags an audio file with an acoustic event.
            Args:
                x (torch.Tensor): audio tensor
                window_size (int): size of the window in samples                
        """
        if len(x.shape) == 2: # [B, N]
            B, N = x.shape
            x = torch.nn.functional.pad(x, (0, window_size - N % window_size), 'constant', 0)
            S = x.shape[1] // window_size
            x = x.reshape(-1, window_size) # [B*S, window_size]
        else:
            B, S, N = x.shape
            x = x.reshape(-1, N)
            x = torch.nn.functional.pad(x, (0, window_size - N), 'constant', 0) # [B*S, window_size]
        x = self.preprocess(x)
        x, _ = self.model._forward_impl(x.unsqueeze(1))
        x = x.reshape(B, S, -1)
        output = []
        for b in range(x.shape[0]):
            output_b = []
            for i in range(x.shape[1]):
               output_b.append(self.parse_output(x[b, i]))
            output.append(output_b)
        return output
    
    def parse_output(self, preds):
        preds = torch.sigmoid(preds.float()).squeeze().detach().cpu().numpy()
        sorted_indexes = np.argsort(preds)[::-1]
        tags =  {'label': [self.labels[sorted_indexes[k]] for k in range(3)],
                # 'probability': [float(preds[sorted_indexes[k]]) for k in range(3)]
                }
        return tags