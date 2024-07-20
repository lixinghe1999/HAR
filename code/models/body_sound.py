import torch.nn as nn
from models.audio_models import Mobilenet_Encoder
from models.imu_models import TransformerEncoder

import torch
import numpy as np
class ClipLoss(nn.Module):
    def __init__(
            self,
            cache_labels=True,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            nn.functional.cross_entropy(logits_per_image, labels) +
            nn.functional.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss

class Body_Sound(nn.Module):
    '''
    A model that map audio and imu to the same space, note that we will lose some information
    '''
    def __init__(self, sequence=0, cfg=[]):
        super().__init__()
        self.audio_model = Mobilenet_Encoder('mn10_as')
        self.imu_model = TransformerEncoder()
        self.proj_audio = nn.Linear(960, 384)
        self.loss = ClipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if 'context' in cfg:
            self.context_audio_model = Mobilenet_Encoder('mn10_as')
            self.proj_context = nn.Linear(960, 384)
            self.context = nn.Linear(768, 91)
    
        if sequence > 0:
            self.sequence_model = nn.LSTM(768, 768, 1, batch_first=True)
        self.fc = nn.Linear(768, 91)
    def freeze_body_sound(self):
        for param in self.proj_audio.parameters():
            param.requires_grad = False
        for param in self.imu_model.parameters():
            param.requires_grad = False
    def forward_contrastive(self, data, train=False, sequence=0, device='cuda'):
        '''
        input: audio + imu
        output: train, CLIP Loss, test, audio, imu feature
        '''
        audio, imu = data['audio'].to(device), data['imu'].to(device)
        if sequence > 0: # if sequence > 0, we will flatten the input
            audio = audio.view(-1, audio.shape[-1]) # [B, S, D] -> [B*S, D]
            imu = imu.view(-1, *imu.shape[-2:]) # [B, S, C, D] -> [B*S, C, D]
        _, audio_feature = self.audio_model(audio)
        audio = self.proj_audio(audio_feature)
        imu = self.imu_model(imu)
        if train:
            return self.loss(audio, imu, self.logit_scale)
        else:
            return audio, imu

    def forward(self, data, train=False, sequence=0, device='cuda'):
        audio, imu = data['audio'].to(device), data['imu'].to(device)
        if sequence > 0: # if sequence > 0, we will flatten the input
            audio = audio.view(-1, audio.shape[-1]) # [B, S, D] -> [B*S, D]
            imu = imu.view(-1, *imu.shape[-2:]) # [B, S, C, D] -> [B*S, C, D]
        _, audio_feature = self.audio_model(audio)
        audio = self.proj_audio(audio_feature)
        imu = self.imu_model(imu)
        if sequence > 0:
            audio = audio.view(-1, sequence, audio.shape[-1])
            imu = imu.view(-1, sequence, imu.shape[-1])  
            feature = torch.cat([audio, imu], dim=2) 
            feature, _ = self.sequence_model(feature)
        else:
            feature = torch.cat([audio, imu], dim=1)
        output = self.fc(feature); scenario = data['scenario'].to(device)

        if train:
            loss = nn.functional.cross_entropy(output, scenario)
            return loss
        else:
            return output, scenario

    def match_eval(self, audio, imu, return_index=False):
        '''
        evaluate the match accuracy, note that the accuracy dependes on batch_size
        '''
        logit_per_audio = audio @ imu.T
        argmax_per_audio = logit_per_audio.argmax(dim=1)
        # print(argmax_per_audio)
        audio_match_mask = (argmax_per_audio == torch.arange(len(argmax_per_audio), device=argmax_per_audio.device))
        audio_match_acc = audio_match_mask.float().mean().item()

        logit_per_imu = imu @ audio.T
        argmax_per_imu = logit_per_imu.argmax(dim=1)
        imu_match_mask = (argmax_per_imu == torch.arange(len(argmax_per_imu), device=argmax_per_imu.device))
        imu_match_acc = imu_match_mask.float().mean().item()
        if return_index:
            return audio_match_acc, imu_match_acc, audio_match_mask, imu_match_mask
        else:
            return audio_match_acc, imu_match_acc