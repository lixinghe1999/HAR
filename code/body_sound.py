from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.body_sound import Body_Sound
import numpy as np
import datetime
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
def body_sound(train_loader, test_loader, model, log_dir, args):
    if args.train:
        best_loss = 100
        for e in range(0, Epoch):
            pbar = tqdm(train_loader)
            train_loss = 0
            model.train()
            for i, data in enumerate(pbar):
                audio, imu = data['audio'].to('cuda'), data['imu'].to('cuda')
                optimizer.zero_grad()
                audio_output, imu_output = model(audio, imu)
                loss = model.loss(audio_output, imu_output, model.logit_scale)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                average_loss = train_loss / (i + 1)
                pbar.set_description(f'Loss: {round(loss.item(), 4)}({round(average_loss, 4)})')
            print(f'Epoch {e} train_loss: {average_loss}')
            if average_loss < best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), '{}/body_sound_{}.pth'.format(log_dir, e))
                print('saved ckpt at epoch {}'.format(e))
    model.eval()
    with torch.no_grad():
        audio_acc, imu_acc = 0, 0
        embeddings = {'audio':[], 'imu': [],}
        for i, data in enumerate(tqdm(test_loader)): 
            if i > 100:
                break
            audio, imu = data['audio'].to('cuda'), data['imu'].to('cuda')
            audio_output, imu_output = model(audio, imu)
            embeddings['audio'].append(audio_output.cpu().numpy())
            embeddings['imu'].append(imu_output.cpu().numpy())
        for key in embeddings:
            embeddings[key] = np.concatenate(embeddings[key], axis=0)

    num_samples = len(embeddings['audio'])
    for b in [4, 8, 16, 32, 64]:
        audio_acc, imu_acc = 0, 0
        for i in range(0, num_samples, b):
            audio = torch.tensor(embeddings['audio'][i:i+b])
            imu = torch.tensor(embeddings['imu'][i:i+b])       
            audio_match_acc, imu_match_acc = model.match_eval(audio, imu, return_index=False)
            audio_acc += audio_match_acc
            imu_acc += imu_match_acc
        average_audio_acc = round(audio_acc / (num_samples // b), 3)
        average_imu_acc = round(imu_acc / (num_samples // b), 3)
        print(f'batch size: {b}, audio_acc: {average_audio_acc}, imu_acc: {average_imu_acc}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    model = Body_Sound().to('cuda')
    if args.log is None:
        log_dir = 'resources/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        print('log_dir: {}'.format(log_dir))
    else:
        log_dir = 'resources/{}'.format(args.log)
        ckpt = torch.load('resources/{}/{}.pth'.format(args.log, args.ckpt))
        model.load_state_dict(ckpt, strict=False)
        print('successfully loaded ckpt from {}'.format(args.log))
    dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_audio_imu.json', window_sec=2, modal=['audio', 'imu'])
    # dataset.save_json('resources/ego4d_audio_imu.json')

    train_size, test_size = int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], 
                                                                generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

    if args.train:
        Epoch = 10
    else:
        Epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    body_sound(train_loader, test_loader, model, log_dir, args)

   