import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification

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
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        
    def forward(self, input_tensor):
        input_tensor = nn.functional.group_norm(input_tensor, 2)
        x = self.linear_embedding(input_tensor)
        print(x.shape)
        # x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        # x = x.permute(1,0,2)[:, -1]
        x = x[:, -1]
        return x 
map_modality = {
    'imu': IMU,
    'camera_pose': Camera_Pose,
    'audio': NotImplementedError
}
class EgoExo4D_Motion(nn.Module): # baseline by Ego-Exo4D
    def __init__(self, embed_dim=256, modality=['imu']):
        super(EgoExo4D_Motion, self).__init__()
        self.encoder = {}
        self.modality = modality
        for m in self.modality:
           setattr(self, m, map_modality[m](embed_dim=embed_dim))
        total_embed_dim = len(modality) * embed_dim
        self.joint_names = ['head', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        # self.joint_names = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']
        self.stabilizer = nn.Sequential(
                        nn.Linear(total_embed_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, len(self.joint_names))
        )
        # self.metric = torchmetrics.classification.MultilabelAccuracy(num_labels=17, average=None)
        self.metric = torchmetrics.F1Score(task="multilabel", num_labels=len(self.joint_names), average=None)
        # pr_auc = torchmetrics.AUC(reorder=True)
        self.loss_func = nn.BCELoss()
        self.regression = True
        
    def forward(self, data, device):
        embeddings = []
        for m in self.modality:
            data[m] = data[m].to(device)
            embedding = getattr(self, m)(data[m])
            embeddings.append(embedding)
        x = torch.cat(embeddings, dim=1)
        global_orientation = self.stabilizer(x)
        return global_orientation
    
    def label_convert(self, target, visible, threshold=1):
        target_motion = torch.diff(target, dim=1).abs().mean(dim=(3))
        target_motion = (target_motion * visible[:, 1:]).mean(dim=(1))

        if len(self.joint_names) == 5:
            joint_cluster = [[0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16]]
            target_motion_new = torch.zeros(target_motion.shape[0], len(joint_cluster), device=target_motion.device, dtype=target_motion.dtype)
            for idx, cluster in enumerate(joint_cluster):
                target_motion_new[:, idx] = target_motion[:, cluster].mean(dim=(1))
            target_motion = target_motion_new

        if self.regression:
            return target_motion
        else:
            motion_mask = target_motion > threshold * target_motion.mean()
            target_motion[motion_mask] = 1
            target_motion[~motion_mask] = 0
            return target_motion
        
    def loss(self, output, target_skeleton, visible,):
        '''
        We only estimate whether the joint has significant motion
        '''
        target_motion = self.label_convert(target_skeleton, visible)
        if self.regression:
            loss = nn.functional.mse_loss(output, target_motion)
        else:
            output = torch.sigmoid(output)
            loss = self.loss_func(output, target_motion)
        return loss
    

    def evaluate(self, output, target_skeleton, visible, test_error,):
        target_motion = self.label_convert(target_skeleton, visible)
        if self.regression:
            error = torch.abs(output - target_motion).mean(dim=(0))
            for idx, joint_name in enumerate(self.joint_names):
                test_error[joint_name].append(error[idx].item())
        else:
            # multi-label classification accuracy
            output = torch.sigmoid(output)
            acc = self.metric(output, target_motion)
            # target = 1 - target_motion.mean(dim=(0))
            # print(acc, target)
            for idx, joint_name in enumerate(self.joint_names):
                test_error[joint_name].append(acc[idx].item())
        return test_error
 