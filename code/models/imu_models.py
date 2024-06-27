# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

from numpy import size
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        # factorized embedding
        self.lin = nn.Linear(cfg['feature_num'], cfg['hidden'])
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg['seq_len'], cfg['hidden']) # position embedding
        else:
            self.pos_embed = pos_embed
        self.norm = nn.LayerNorm(cfg['hidden'])
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)
    
class TransformerEncoder(torch.nn.Module):
    def __init__(self, in_channels=6, size_embeddings=384):
        super().__init__()
        self.name = TransformerEncoder
        self.norm = torch.nn.GroupNorm(2, 6)
        self.fc = nn.Conv1d(
                in_channels=in_channels,
                out_channels=size_embeddings,
                kernel_size=10,
                dilation=2,
            )
        # self.fc = Embeddings({'feature_num': size_embeddings, 'hidden': size_embeddings, 'seq_len': 100})

        encoder_layer = nn.TransformerEncoderLayer(d_model=size_embeddings, nhead=4, dim_feedforward=size_embeddings, batch_first=True)
        self.model = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
    def forward(self, batch):
        batch = self.norm(batch)
        batch = self.fc(batch)
        batch = batch.permute(0, 2, 1)
        batch = self.model(batch)[:, 0, :]
        return batch
    

