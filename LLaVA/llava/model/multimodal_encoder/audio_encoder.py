import torch
import torch.nn as nn
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from utils import NAME_TO_WIDTH
from models.preprocess import AugmentMelSTFT

def audio_backbone(model_name, strides, head_type):
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                strides=strides)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                strides=strides, head_type=head_type)
    return model

class AUDIO(nn.Module):
    def __init__(self, model_name, strides, head_type):
        super(AUDIO, self).__init__()
        self.model = audio_backbone(model_name, strides, head_type)
        self.mel = AugmentMelSTFT(n_mels=64, sr=48000, win_length=1024, hopsize=256)

    def forward(self, waveform):
        spec = self.mel(waveform)
        preds, features = self.model(spec.unsqueeze(0))
        preds = torch.sigmoid(preds.float()).squeeze().detach().cpu().numpy()
        return preds, features

if __name__ == '__main__':
    model_name = 'dymn10_as'
    strides = [2, 2, 2, 2]
    head_type = 'mlp'

    model = AUDIO(model_name, strides, head_type)
    audio = torch.zeros(1, 48000)
    output, features = model(audio)
    print(output.shape, features.shape)
    print('Done!')