from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration, Ego4D_Narration_Sequence
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

def scenario(train_loader, test_loader, model, log_dir, args):
    if args.train:
        best_loss = 100
        for e in range(0, Epoch):
            pbar = tqdm(train_loader)
            train_loss = 0
            model.train()
            for i, data in enumerate(pbar):
                optimizer.zero_grad()
                loss = model(data, train=True, sequence=args.sequence)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                average_loss = train_loss / (i + 1)
                pbar.set_description(f'Loss: {round(loss.item(), 4)}({round(average_loss, 4)})')
            print(f'Epoch {e} train_loss: {average_loss}')
            if average_loss < best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), '{}/scenario_{}.pth'.format(log_dir, e))
                print('saved ckpt at epoch {}'.format(e))
    model.eval()
    with torch.no_grad():
        preds, gts = [], []
        for i, data in enumerate(tqdm(test_loader)): 
            if i > 100:
                break
            pred, gt = model(data, train=False, sequence=args.sequence)
            preds.append(pred.cpu()); gts.append(gt.cpu())
        preds = torch.cat(preds, dim=0).argmax(dim=1)
        gts = torch.cat(gts, dim=0).argmax(dim=1)
    from torchmetrics import Accuracy
    accuracy = Accuracy(task='multiclass', num_classes=91)
    acc = accuracy(preds, gts).item()
    print(acc)
def body_sound(train_loader, test_loader, model, log_dir, args):
    if args.train:
        best_loss = 100
        for e in range(0, Epoch):
            pbar = tqdm(train_loader)
            train_loss = 0
            model.train()
            for i, data in enumerate(pbar):
                optimizer.zero_grad()
                loss = model.forward_contrastive(data, train=True, sequence=args.sequence)
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
        embeddings = {'audio':[], 'imu': [], 'cosine': []}
        for i, data in enumerate(tqdm(test_loader)): 
            if i > 20:
                break
            audio_output, imu_output = model.forward_contrastive(data, train=False, sequence=args.sequence)
            embeddings['audio'].append(audio_output.cpu().numpy())
            embeddings['imu'].append(imu_output.cpu().numpy())
            if 'cosine' in data:
                embeddings['cosine'].append(data['cosine'])
            else:
                embeddings['cosine'].append(0)
        for key in embeddings:
            embeddings[key] = np.concatenate(embeddings[key], axis=0)
        # step1, evaluate by pair-to-pair retrieval
        retrieval(embeddings, model)
        # step2, evaluate by match with text and audio
        cosine_match(embeddings)
def cosine_match(embeddings):
    cosine_audio_imu = np.dot(embeddings['audio'], embeddings['imu'].T) / np.linalg.norm(embeddings['audio']) / np.linalg.norm(embeddings['imu'])
    cosine_audio_imu = np.diag(cosine_audio_imu)
    cosine_gt = embeddings['cosine']
    print(cosine_audio_imu.shape, cosine_gt.shape)
    # check whether the curve match?

    simi = np.dot(cosine_audio_imu, cosine_gt.T) / np.linalg.norm(cosine_audio_imu) / np.linalg.norm(cosine_gt)
    
    # fit a curve to the cosine_audio_imu
    coefficients = np.polyfit(cosine_audio_imu, cosine_gt, deg=2)
    print("Polynomial coefficients:", coefficients)
    x_fit = np.linspace(cosine_audio_imu.min(), cosine_audio_imu.max(), 100)
    # Compute the y-values for the fitted curve
    y_fit = np.polyval(coefficients, x_fit)
    plt.scatter(cosine_audio_imu, cosine_gt, label='cosine_audio_imu')
    plt.plot(x_fit, y_fit, label='fit')
    plt.savefig('figs/cosine_match.png')

    # plt.figure()
    # plt.scatter(range(len(cosine_audio_imu)), cosine_audio_imu/np.max(cosine_audio_imu), label='cosine_audio_imu')


    # plt.plot(cosine_audio_imu/np.max(cosine_audio_imu), label='cosine_audio_imu')
    # plt.plot(cosine_gt/np.max(cosine_gt), label='cosine_gt')
    # plt.legend()
    # plt.savefig('figs/cosine_match.png')
    # print('similarity between cosine_audio_imu and cosine_gt: {}'.format(simi))
    
def retrieval(embeddings, model):
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
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--ckpt', type=str,)
    parser.add_argument('--sequence', type=int, default=0)
    parser.add_argument('--mode', type=str, default='body_sound')
    args = parser.parse_args()
    model = Body_Sound(sequence=args.sequence).to('cuda')
    if args.log is None:
        log_dir = 'resources/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        print('log_dir: {}'.format(log_dir))
    else:
        log_dir = 'resources/{}'.format(args.log)
        ckpt = torch.load('resources/{}/{}.pth'.format(args.log, args.ckpt))
        model.load_state_dict(ckpt, strict=False)
        print('successfully loaded ckpt from {}'.format(args.log))

    dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_narration_prune.json', window_sec=2, modal=['audio', 'imu']) 
    if args.sequence > 0:
        train_dataset = Ego4D_Narration_Sequence(dataset, args.sequence)

    train_idx, test_idx = dataset.split_with_scenario(ratio=0.8)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

    if args.train:
        Epoch = 5
    else:
        Epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if args.mode == 'scenario':
        scenario(train_loader, test_loader, model, log_dir, args)
    elif args.mode == 'body_sound':
        body_sound(train_loader, test_loader, model, log_dir, args)

   