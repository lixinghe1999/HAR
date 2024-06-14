# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

from numpy import size
import torch
import torch.nn as nn

class MW2StackRNNPooling(torch.nn.Module):
    def __init__(self, input_dim=32, size_embeddings: int = 128):
        super().__init__()
        self.name = MW2StackRNNPooling
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(2, 6),
            Block(6, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5),
            torch.nn.GroupNorm(4, input_dim),
            torch.nn.GRU(
                batch_first=True, input_size=input_dim, hidden_size=size_embeddings
            ),
        )

    def forward(self, batch):
        # return the last hidden state
        return self.net(batch)[1][0]

class TransformerEncoder(torch.nn.Module):
    def __init__(self, in_channels=6, size_embeddings=128):
        super().__init__()
        self.name = TransformerEncoder
        self.norm = torch.nn.GroupNorm(2, 6)
        self.fc = nn.Conv1d(
                in_channels=in_channels,
                out_channels=size_embeddings,
                kernel_size=10,
                dilation=2,
            )
        encoder_layer = nn.TransformerEncoderLayer(d_model=size_embeddings, nhead=4, dim_feedforward=size_embeddings)
        self.model = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
    def forward(self, batch):
        batch = self.norm(batch)
        batch = self.fc(batch)
        # the CLS token is the first token
        batch = batch.permute(0, 2, 1)
        cls = torch.zeros(batch.shape[0], 1, batch.shape[2]).to(batch.device)
        batch = torch.cat((cls, batch), dim=1)
        batch = self.model(batch)[:, 0, :]
        return batch
    

