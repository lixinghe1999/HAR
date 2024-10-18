
from ego4d.ego4d_dataset import Ego4D_IMU2CLIP
from models.imu_models import TransformerEncoder
import pytorch_lightning as pl
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torchmetrics


class Coarse_grained_localization(pl.LightningModule):
    '''
    Localize by region-wise classification
    '''
    def __init__(self,  lr=1e-3):
        super().__init__()
        dataset =  Ego4D_IMU2CLIP(folder='../dataset/ego4d/v2/', window_sec=5, modal=['imu'])

        self.train_dataset, self.test_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

        print('number of training samples: ', len(self.train_dataset), 'number of testing samples: ', len(self.test_dataset))
        self.model = nn.Sequential(
            TransformerEncoder(),
            nn.Linear(384, 4)
        )
        self.lr = lr
        self.accuracy = torchmetrics.Precision(task='multiclass', num_classes=4, average='micro')
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['imu'], batch['scenario']
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['imu'], batch['scenario']
        y_hat = self.model(x)
        acc = self.accuracy(y_hat, torch.argmax(y, dim=1))
        self.log('accuracy', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        return nn.CrossEntropyLoss()(y_hat, torch.argmax(y, dim=1))
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=4)  

if __name__ == "__main__":
    model = Coarse_grained_localization()
    trainer = pl.Trainer(max_epochs=50, devices=1)
    trainer.fit(model)