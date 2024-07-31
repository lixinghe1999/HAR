import torch.nn as nn
import torch
class SequenceModel_Transformer(nn.Module):
    def __init__(self, input_size=768, layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4, dim_feedforward=input_size, batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=layers)
    def forward(self, x):
        # get the causal mask for transformer encoder
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).to(x.device)
        x = self.model(x, mask=mask.bool())
        if self.training:
            weight_along_time = torch.arange(1, x.size(1)+1, device=x.device).float().unsqueeze(0)
            weight_along_time = weight_along_time / weight_along_time.sum()
            x = x * weight_along_time.unsqueeze(-1)
            return x.mean(dim=1)

        else:
            return x[:, -1]

class SequenceModel_LSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, layers=1):
        super().__init__()
        self.model = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
    def forward(self, x):
        x, _ = self.model(x)
        return x[:, -1]
        
        