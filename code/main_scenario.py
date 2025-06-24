from ego4d.ego4d_dataset import Ego4D_Narration
from tqdm import tqdm
from models.multi_modal import Multi_modal_model
import numpy as np
import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
import torchmetrics

class ScenarioRecognitionModel(pl.LightningModule):
    def __init__(self, num_class, sequence=0, modality_mask=None, eval_type='multi-label', mode='scenario'):
        super().__init__()
        self.model = Multi_modal_model(sequence=sequence, num_class=num_class)
        self.num_class = num_class
        self.sequence = sequence
        self.modality_mask = modality_mask
        self.eval_type = eval_type
        self.mode = mode
        self.save_hyperparameters()

        # Initialize metrics for scenario mode
        if self.mode == 'scenario':
            self.loss = torch.nn.CrossEntropyLoss()
            if self.eval_type == 'multi-label':
                self.val_metric = torchmetrics.F1Score(task='multilabel', num_labels=num_class)
            else:  # multi-class
                self.val_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)

        # Lists to store predictions, ground truths, or embeddings
        self.preds = []
        self.gts = []
        self.embeddings = {'audio': [], 'imu': [], 'scenario': []}

    def training_step(self, batch, batch_idx):
        if self.mode == 'scenario':
            output = self.model(batch, sequence=self.sequence, modality_mask=self.modality_mask)
            loss = self.loss(output, batch['scenario'])
        else:  # body_sound
            loss = self.model.forward_contrastive(batch, train=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == 'scenario':
            output = self.model(batch, sequence=self.sequence, modality_mask=self.modality_mask)
            acc = self.val_metric(output, batch['scenario'])
            self.log(f'{self.eval_type}_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        else:  # body_sound
            audio_output, imu_output = self.model.forward_contrastive(batch, train=False)
            self.embeddings['audio'].append(audio_output.cpu().numpy())
            self.embeddings['imu'].append(imu_output.cpu().numpy())
            self.embeddings['scenario'].append(batch['scenario'].cpu().numpy())

    def on_validation_epoch_end(self):
        if self.mode == 'body_sound':
            for key in self.embeddings:
                self.embeddings[key] = np.concatenate(self.embeddings[key], axis=0)
            np.savez(f'{self.trainer.log_dir}/embeddings.npz', **self.embeddings)
            self._retrieval()
            self.embeddings = {'audio': [], 'imu': [], 'scenario': []}
        super().on_validation_epoch_end()

    def _retrieval(self):
        def norm_cosine_similarity_matrix(a, b):
            a = torch.tensor(a)
            b = torch.tensor(b)
            dot_product = a @ b.T
            norm_a = torch.norm(a, dim=1).unsqueeze(1)
            norm_b = torch.norm(b, dim=1).unsqueeze(0)
            norm_product = norm_a @ norm_b
            output = dot_product / norm_product
            return output.mean().item()

        num_samples = len(self.embeddings['audio'])
        for b in [4, 8, 16, 32, 64]:
            audio_acc, imu_acc = 0, 0
            for i in range(0, num_samples, b):
                audio = torch.tensor(self.embeddings['audio'][i:i+b])
                imu = torch.tensor(self.embeddings['imu'][i:i+b])
                audio_match_acc, imu_match_acc = self.model.match_eval(audio, imu, return_index=False)
                audio_acc += audio_match_acc
                imu_acc += imu_match_acc
            average_audio_acc = round(audio_acc / (num_samples // b), 3)
            average_imu_acc = round(imu_acc / (num_samples // b), 3)
            self.log(f'batch_size_{b}_audio_acc', average_audio_acc, on_epoch=True)
            self.log(f'batch_size_{b}_imu_acc', average_imu_acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ego4d_narration')
    parser.add_argument('--sequence', type=int, default=0)
    parser.add_argument('--mode', type=str, default='scenario') # body_sound - contrastive learning, scenario - classification
    parser.add_argument('--epoch', type=int, default=1) # if train = False, epoch is set to 0
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--window_sec', type=int, default=30)
    parser.add_argument('--eval', type=str, default='multi-label', choices=['multi-label', 'multi-class']) 
    parser.add_argument('--num_class', type=int, default=91)
    parser.add_argument('--modality_mask', type=str, default=None, choices=['audio', 'imu'])
    args = parser.parse_args()

    model = ScenarioRecognitionModel(
        num_class=args.num_class,
        sequence=args.sequence,
        modality_mask=args.modality_mask,
        eval_type=args.eval,
        mode=args.mode
    )
    logger = loggers.TensorBoardLogger('resources/scenario', default_hp_metric=False)
    trainer = pl.Trainer(max_epochs=args.epoch, logger=logger, devices=[0], accelerator='cuda', enable_progress_bar=True)

    # Set up log directory
    if args.ckpt is not None:
        model = model.load_from_checkpoint(f'resources/{args.ckpt}.pth', strict=False)

    dataset = Ego4D_Narration(window_sec=args.window_sec, modal=['audio', 'imu']) 

    train_idx, test_idx = dataset.split_with_scenario(ratio=0.8)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    if args.ckpt:
        ckpt_path = f'resources/{args.log}/{args.ckpt}.pth'
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint, strict=False)
            print(f'Successfully loaded checkpoint from {ckpt_path}')
        else:
            print(f'Checkpoint {ckpt_path} not found')
            exit()

    if args.train:
        trainer.fit(model, train_loader, test_loader)
    else:
        trainer.validate(model, test_loader)
   
   