from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.body_sound import Body_Sound
import numpy as np
import datetime
import argparse
import os

import torch
def body_sound(train_loader, test_loader, model, log_dir, args):
    if args.train:
        best_loss = 100
        for e in range(0, Epoch):
            pbar = tqdm(train_loader)
            train_loss = 0
            model.train()
            for i, data in enumerate(pbar):
                optimizer.zero_grad()
                loss = model.forward_context(data, train=True, device='cuda')
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                average_loss = train_loss / (i + 1)
                pbar.set_description(f'Loss: {round(loss.item(), 4)}({round(average_loss, 4)})')
            print(f'Epoch {e} train_loss: {average_loss}')
            if average_loss < best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), '{}/context_{}.pth'.format(log_dir, e))
                print('saved ckpt at epoch {}'.format(e))
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader)
        num_samples = len(test_loader)
        nun_samples = 100
        embeddings = {'pred':[], 'gt':[]}
        for i, data in enumerate(pbar):
            if i > nun_samples:
                break
            output, gt = model.forward_context(data, train=False, device='cuda')
            embeddings['pred'].append(output.cpu())
            embeddings['gt'].append(gt.cpu())
        for key in embeddings:
            embeddings[key] = torch.cat(embeddings[key], dim=0)
    from torchmetrics import Accuracy
    accuracy = Accuracy(task='multiclass', num_classes=91)
    print(accuracy(embeddings['pred'], embeddings['gt']).item())
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, default='body_sound_9')
    args = parser.parse_args()

    model = Body_Sound().to('cuda')
    model.freeze_body_sound()
    if args.log is None:
        log_dir = 'resources/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        print('log_dir: {}'.format(log_dir))
    else:
        log_dir = 'resources/{}'.format(args.log)
        ckpt = torch.load('resources/{}/{}.pth'.format(args.log, args.ckpt))
        model.load_state_dict(ckpt, strict=False)
        print('successfully loaded ckpt from {}'.format('resources/{}/{}.pth'.format(args.log, args.ckpt)))
    
    dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_audio_imu.json', modal=['audio', 'imu'])
    # dataset.save_json('resources/ego4d_audio_imu.json')

    train_idx, test_idx = dataset.split_with_scenario(ratio=0.8)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

    if args.train:
        Epoch = 2
    else:
        Epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    body_sound(train_loader, test_loader, model, log_dir, args)

   