import torch.utils
import torch.utils.data
from models.multi_modal import Multi_modal_model
from utils.custom_dataset import CustomDataset
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


# example class for pytorch lightning
class SAMoSA(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = Multi_modal_model(num_class=13)
        dataset = CustomDataset('../dataset/aiot/Lixing_home-20241106_082431_132')
        train_data, test_data = dataset.__split__()
        self.train_dataset = torch.utils.data.Subset(dataset, train_data)
        self.test_dataset = torch.utils.data.Subset(dataset, test_data)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=13)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        acc = self.accuracy(y_hat, y)
        self.log('accuracy', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss(self, y_hat, y):
        return nn.CrossEntropyLoss()(y_hat, y)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=True)
    
if __name__ == '__main__':
    from pytorch_lightning.loggers import CSVLogger

    model = SAMoSA()
    csv_logger = CSVLogger(save_dir='lightning_logs', name="samosa")

    trainer = pl.Trainer(logger=csv_logger, max_epochs=10, devices=1, log_every_n_steps=5)
    trainer.fit(model)