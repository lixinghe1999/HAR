import torch.nn as nn

class IMU(nn.Module):
    def __init__(self, input_dim=100, num_layer=3, embed_dim=256, nhead=8,):
        super(IMU, self).__init__()
        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        
    def forward(self, input_tensor):
        input_tensor = nn.functional.group_norm(input_tensor, 2, 12)
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)[:, -1]
        return x 