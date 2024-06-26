import torch.nn as nn
from models.audio_models import Mobilenet_Encoder
from models.imu_models import TransformerEncoder
import torch
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

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            nn.functional.cross_entropy(logits_per_image, labels) +
            nn.functional.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
class Body_Sound(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.audio_model = Mobilenet_Encoder()
        self.imu_model = TransformerEncoder()
        self.proj_audio = nn.Linear(3840, 384)
        self.clip_loss = ClipLoss()
    def forward(self, audio, imu):
        audio = self.audio_model(audio)
        audio = self.proj_audio(audio)
        imu = self.imu_model(imu)
        return audio, imu
    def match_eval(self, audio, imu, logit_scale=1.0):
        logit_per_audio = audio @ imu.T
        argmax_per_audio = logit_per_audio.argmax(dim=1)
        audio_match_accuracy = (argmax_per_audio == torch.arange(len(argmax_per_audio), device=argmax_per_audio.device)).float().mean()

        logit_per_imu = imu @ audio.T
        argmax_per_imu = logit_per_imu.argmax(dim=1)
        imu_match_accuracy = (argmax_per_imu == torch.arange(len(argmax_per_imu), device=argmax_per_imu.device)).float().mean()

        return audio_match_accuracy, imu_match_accuracy
    