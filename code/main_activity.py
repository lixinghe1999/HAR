'''
Activity Recognition for Ego4D
'''

from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Moment
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.multi_modal import Multi_modal_model
import numpy as np
import datetime
import argparse
import os
import torch
import pytorch_lightning as pl
import torchmetrics


class ActivityRecognitionModel(pl.LightningModule):
    def __init__(self, num_class, modality_mask=None, eval_type='single-label'):
        super().__init__()
        self.model = Multi_modal_model(num_class=num_class)
        self.num_class = num_class
        self.modality_mask = modality_mask
        self.eval_type = eval_type
        self.save_hyperparameters()

        # Initialize metrics
        if self.eval_type == 'multi-label':
            self.val_metric = torchmetrics.F1Score(task='multilabel', num_labels=num_class)
        else:
            self.val_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)

        # Lists to store predictions and ground truths
        self.preds = []
        self.gts = []

    def forward(self, x, train=False):
        return self.model(x, train=train, modality_mask=self.modality_mask, target='activity')

    def training_step(self, batch, batch_idx):
        loss = self(batch, train=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, gt = self(batch, train=False)
        self.preds.append(pred.cpu())
        self.gts.append(gt.cpu())

    def on_validation_epoch_end(self):
        preds = torch.cat(self.preds, dim=0)
        gts = torch.cat(self.gts, dim=0)

        if self.eval_type == 'single-label':
            preds = preds.argmax(dim=1)
            # Save predictions and ground truths
            np.save(f'{self.trainer.log_dir}/preds.npy', preds.numpy())
            np.save(f'{self.trainer.log_dir}/gts.npy', gts.numpy())

        acc = self.val_metric(preds, gts.long() if self.eval_type == 'multi-label' else gts)
        self.log(f'{self.eval_type}_acc', acc, on_epoch=True, prog_bar=True)

        # Clear lists for next epoch
        self.preds = []
        self.gts = []
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ego4d_narration', choices=['ego4d_narration'])
    parser.add_argument('--epoch', type=int, default=3) # if train = False, epoch is set to 0
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--window_sec', type=int, default=10)
    parser.add_argument('--eval', type=str, default='single-label', choices=['multi-label', 'single-label']) 
    parser.add_argument('--num_class', type=int, default=50)
    parser.add_argument('--modality_mask', type=str, default=None, choices=['audio', 'imu'])
    args = parser.parse_args()

    model = ActivityRecognitionModel(
        num_class=args.num_class,
        modality_mask=args.modality_mask,
        eval_type=args.eval
    )
    trainer = pl.Trainer(max_epochs=args.epoch if args.train else 0, default_root_dir='resources/activity', devices=[0])
    # Set up log directory

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print(f'Load model from {args.ckpt}')

    dataset = Ego4D_Narration(window_sec=args.window_sec, modal=['audio', 'imu', 'capture24']) 

    train_idx, test_idx = dataset.split_with_scenario(ratio=0.8)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    

    if args.train:
        trainer.fit(model, train_loader, test_loader)
    else:
        trainer.validate(model, test_loader)
   