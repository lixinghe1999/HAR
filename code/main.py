'''
Scenario Recognition for Ego4D and EgoExo4D
'''

from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Narration_Sequence, Ego4D_Free, Ego4D_Sound
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.body_sound import Body_Sound
import numpy as np
import datetime
import argparse
import os
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
                loss = model(data, train=True, sequence=args.sequence, modality_mask=args.modality_mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                average_loss = train_loss / (i + 1)
                pbar.set_description(f'Loss: {round(loss.item(), 4)}({round(average_loss, 4)})')
            print(f'Epoch {e} train_loss: {average_loss}')
            if average_loss < best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), '{}/scenario_{}_{}.pth'.format(log_dir, args.sequence, e))
                print('saved ckpt at epoch {}'.format(e))
    model.eval()
    with torch.no_grad():
        preds, gts = [], []
        for i, data in enumerate(tqdm(test_loader)): 
            if i > 200:
                break
            pred, gt = model(data, train=False, sequence=args.sequence, modality_mask=args.modality_mask)
            # print(torch.sigmoid(pred), gt)
            preds.append(pred.cpu()); gts.append(gt.cpu())
        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
    # Evaluation
    import torchmetrics
    # multi-label classification
    if args.eval == 'multi-label':
        accuracy = torchmetrics.F1Score(task='multilabel', num_labels=args.num_class)
        acc = accuracy(preds, gts.long())
    elif args.eval == 'multi-class':
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class)
        preds = preds.argmax(dim=1); gts = gts.argmax(dim=1)
        acc = accuracy(preds, gts).item()
    print(f'The {args.eval} evaluation results is {acc}')
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
        embeddings = {'audio':[], 'imu': [], 'scenario': []}
        for i, data in enumerate(tqdm(test_loader)): 
            # if i > 5:
            #     break
            audio_output, imu_output = model.forward_contrastive(data, train=False, sequence=args.sequence)
            embeddings['audio'].append(audio_output.cpu().numpy())
            embeddings['imu'].append(imu_output.cpu().numpy())
            embeddings['scenario'].append(data['scenario'].cpu().numpy())
        for key in embeddings:
            embeddings[key] = np.concatenate(embeddings[key], axis=0)
        retrieval(embeddings, model)
def retrieval(embeddings, model):
    def norm_cosine_similarity_matrix(a, b):
        dot_product = a @ b.T
        norm_a = torch.norm(a, dim=1).unsqueeze(1)
        norm_b = torch.norm(b, dim=1).unsqueeze(0)
        norm_product = norm_a @ norm_b
        output = dot_product / norm_product
        return output.mean().item()
    num_samples = len(embeddings['audio'])

    # Batch-evaluation, intuition and convenience
    # for b in [4, 8, 16, 32, 64]:
    #     audio_acc, imu_acc = 0, 0
    #     for i in range(0, num_samples, b):
    #         audio = torch.tensor(embeddings['audio'][i:i+b])
    #         imu = torch.tensor(embeddings['imu'][i:i+b])       
    #         audio_match_acc, imu_match_acc = model.match_eval(audio, imu, return_index=False)
    #         audio_acc += audio_match_acc
    #         imu_acc += imu_match_acc
    #     average_audio_acc = round(audio_acc / (num_samples // b), 3)
    #     average_imu_acc = round(imu_acc / (num_samples // b), 3)
    #     print(f'batch size: {b}, audio_acc: {average_audio_acc}, imu_acc: {average_imu_acc}')

    # split embeddings by scenario
    scenario_idx = {}
    for i in range(91):
        scenario_idx[i] = []
        for j in range(num_samples):
            if embeddings['scenario'][j, i] == 1:
                scenario_idx[i].append(j)
    diffs = []
    for i in range(91):
        select_idx = scenario_idx[i]
        remain_idx = list(set(range(num_samples)) - set(select_idx))
        if len(select_idx) == 0 or len(remain_idx) == 0:
            continue
        audio = torch.tensor(embeddings['audio'][select_idx]); imu = torch.tensor(embeddings['imu'][select_idx])

        # normalize cosine similarity
        average_similarity = norm_cosine_similarity_matrix(audio, imu)
        remain_audio = torch.tensor(embeddings['audio'][remain_idx]); remain_imu = torch.tensor(embeddings['imu'][remain_idx])

        next_average_similarity = (norm_cosine_similarity_matrix(remain_audio, imu) + norm_cosine_similarity_matrix(audio, remain_imu))/2
        diff = average_similarity - next_average_similarity
        diffs.append(diff)
        print('Intra-scenario similarity: {}, Inter-scenario similarity: {}'.format(average_similarity, next_average_similarity))
    print('Average difference: {}'.format(np.mean(diffs)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--log', type=str, default=None) # work on existing log, auto make a new log by time, or work on a specific log
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ego4d_narration', choices=['ego4d_narration', 'ego4d_free', 'egoexo', 'ego4d_sound'])
    parser.add_argument('--sequence', type=int, default=0)
    parser.add_argument('--mode', type=str, default='scenario') # body_sound - contrastive learning, scenario - classification
    parser.add_argument('--epoch', type=int, default=5) # if train = False, epoch is set to 0
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--window_sec', type=int, default=2)
    parser.add_argument('--eval', type=str, default='multi-label', choices=['multi-label', 'multi-class']) 
    parser.add_argument('--num_class', type=int, default=91)
    parser.add_argument('--modality_mask', type=str, default=None, choices=['audio', 'imu'])
    args = parser.parse_args()
    model = Body_Sound(sequence=args.sequence, num_class=args.num_class).to('cuda')
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
        dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_narration_prune.json', window_sec=args.window_sec, modal=['audio', 'imu']) 
        if args.sequence > 0:
            dataset = Ego4D_Narration_Sequence(dataset, args.sequence)
    elif args.dataset == 'ego4d_sound':
        dataset = Ego4D_Sound(window_sec=args.window_sec, modal=['audio', 'imu'])
        if args.sequence > 0:
            dataset = Ego4D_Narration_Sequence(dataset, args.sequence)
    elif args.dataset == 'ego4d_free':
        dataset = Ego4D_Free(window_sec=args.window_sec, modal=['audio', 'imu']) # or you can set window_sec = sequence * 2 to make it fair
    elif args.dataset == 'egoexo':
        dataset = EgoExo_atomic(window_sec=args.window_sec, modal=['audio', 'imu'])
        dataset.prune_slience()

    train_idx, test_idx = dataset.split_with_scenario(ratio=0.8)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    if args.train:
        Epoch = args.epoch
    else:
        Epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if args.mode == 'scenario':
        scenario(train_loader, test_loader, model, log_dir, args)
    elif args.mode == 'body_sound':
        body_sound(train_loader, test_loader, model, log_dir, args)

   