'''
Activity Recognition for Ego4D
'''

from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Narration_Sequence, Ego4D_Free, Ego4D_Sound
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.multi_modal import Multi_modal_model
import numpy as np
import datetime
import argparse
import os
import torch

def activity(train_loader, test_loader, model, log_dir, args):
    if args.train:
        best_loss = 100
        for e in range(0, Epoch):
            pbar = tqdm(train_loader)
            train_loss = 0
            model.train()
            for i, data in enumerate(pbar):
                optimizer.zero_grad()
                loss = model(data, train=True, modality_mask=args.modality_mask, target='activity')
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                average_loss = train_loss / (i + 1)
                pbar.set_description(f'Loss: {round(loss.item(), 4)}({round(average_loss, 4)})')
            print(f'Epoch {e} train_loss: {average_loss}')
            if average_loss < best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), '{}/activity_{}.pth'.format(log_dir, e))
                print('saved ckpt at epoch {}'.format(e))
    print('Start evaluation')
    model.eval()
    import time
    t_start = time.time()
    with torch.no_grad():
        preds, gts = [], []
        for i, data in enumerate(tqdm(test_loader)): 
            pred, gt = model(data, train=False, modality_mask=args.modality_mask, target='activity')
            preds.append(pred.cpu()); gts.append(gt.cpu())
        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
    # Evaluation
    import torchmetrics
    # multi-label classification
    if args.eval == 'multi-label':
        accuracy = torchmetrics.F1Score(task='multilabel', num_labels=args.num_class)
        acc = accuracy(preds, gts.long())
    elif args.eval == 'single-label':
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class)
        preds = preds.argmax(dim=1)
        acc = accuracy(preds, gts).item()
        preds_numpy = preds.numpy()
        gts_numpy = gts.numpy()
        # save the results
        np.save('{}/preds.npy'.format(log_dir), preds_numpy)
        np.save('{}/gts.npy'.format(log_dir), gts_numpy)
    print(f'The {args.eval} evaluation results is {acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--log', type=str, default=None) # work on existing log, auto make a new log by time, or work on a specific log
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ego4d_narration', choices=['ego4d_narration'])
    parser.add_argument('--epoch', type=int, default=3) # if train = False, epoch is set to 0
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--window_sec', type=int, default=10)
    parser.add_argument('--eval', type=str, default='single-label', choices=['multi-label', 'single-label']) 
    parser.add_argument('--num_class', type=int, default=50)
    parser.add_argument('--modality_mask', type=str, default=None, choices=['audio', 'imu'])
    args = parser.parse_args()


    model = Multi_modal_model(num_class=args.num_class).to('cuda')
    if args.log is None: # No log and train, create a new log
        if args.train:
            log_dir = 'resources/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            os.makedirs(log_dir, exist_ok=True)
            print('Working on new log_dir: {}'.format(log_dir))
        else:
            print('Please specify a log_dir')
            exit()
    else:
        log_dir = 'resources/{}'.format(args.log)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print('Working on new log_dir: {}'.format(log_dir))
        else:
            if args.ckpt is not None:
                ckpt_path = 'resources/{}/{}.pth'.format(args.log, args.ckpt)
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt, strict=False)
                print('successfully loaded ckpt from {}'.format(ckpt_path))
            else:
                print('Working on existing log_dir: {}'.format(log_dir))
    if args.dataset == 'ego4d_narration':
        dataset = Ego4D_Narration(window_sec=args.window_sec, modal=['audio', 'imu', 'capture24']) 

    train_idx, test_idx = dataset.split_with_scenario(ratio=0.8)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if args.train:
        Epoch = args.epoch
    else:
        Epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    activity(train_loader, test_loader, model, log_dir, args)

   