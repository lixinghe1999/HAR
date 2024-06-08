import torch
import torch.nn as nn


class Camera_Pose(nn.Module):
    def __init__(self, input_dim=40, num_layer=3, embed_dim=256, nhead=8,):
        super(Camera_Pose, self).__init__()
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        
    def forward(self, input_tensor):
        input_tensor = input_tensor.permute(0, 2, 1)
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)[:, -1]
        return x 
class IMU(nn.Module):
    def __init__(self, input_dim=800, num_layer=3, embed_dim=256, nhead=8,):
        super(IMU, self).__init__()
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        
    def forward(self, input_tensor):
        input_tensor = nn.functional.group_norm(input_tensor, 2)
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)[:, -1]
        return x 
map_modality = {
    'imu': IMU,
    'camera_pose': Camera_Pose,
    'audio': NotImplementedError
}
class EgoExo4D_Baseline(nn.Module): # baseline by Ego-Exo4D
    def __init__(self, embed_dim=256, modality=['imu']):
        super(EgoExo4D_Baseline, self).__init__()
        self.encoder = {}
        self.modality = modality
        for m in self.modality:
           setattr(self, m, map_modality[m](embed_dim=embed_dim))
        total_embed_dim = len(modality) * embed_dim

        self.stabilizer = nn.Sequential(
                        nn.Linear(total_embed_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 51)
        )
    def forward(self, data, device):
        embeddings = []
        for m in self.modality:
            data[m] = data[m].to(device)
            embedding = getattr(self, m)(data[m])
            embeddings.append(embedding)
        x = torch.cat(embeddings, dim=1)
        global_orientation = self.stabilizer(x)
        return global_orientation
    
    def loss(self, output, target_skeleton, visible):
        output = output.reshape(-1, 17, 3)
        target_skeleton = target_skeleton[:, -1]
        visible = visible[:, -1]
        loss = ((output - target_skeleton)**2 * visible.unsqueeze(2)).mean()
        return loss


    def evaluate(self, output, target_skeleton, visible, joint_names, test_error):
        output = output.reshape(-1, 17, 3)
        target_skeleton = target_skeleton[:, -1]
        visible = visible[:, -1]
        error_sum = ((output - target_skeleton).abs()* visible.unsqueeze(2)).sum(dim=(0, 2))

        visible_sum = visible.sum(dim=(0))
        mask = visible_sum > 0
        for i, m in enumerate(mask):
            if m:
                joint_name = joint_names[i]
                error = error_sum[i] / visible_sum[i]
                test_error[joint_name].append(error.item())
        return test_error
