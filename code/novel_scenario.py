from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Narration_Sequence, Ego4D_Free
from tqdm import tqdm
from models.body_sound import Body_Sound
from models.audio_models import Mobilenet_Encoder
import numpy as np
import datetime
import argparse
import os
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
                torch.save(model.state_dict(), '{}/scenario_{}_{}.pth'.format(log_dir, args.sequence, e))
                print('saved ckpt at epoch {}'.format(e))
    model.eval()
    with torch.no_grad():
        preds, gts = [], []
        for i, data in enumerate(tqdm(test_loader)): 
            if i > 100:
                break
            pred, gt = model(data, train=False, sequence=args.sequence)
            preds.append(pred.cpu()); gts.append(gt.cpu())
        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
    # Evaluation
    import torchmetrics
    # multi-label classification
    if args.eval == 'multi-label':
        # accuracy = Accuracy(task='multilabel', num_labels=91, threshold=1.0)
        accuracy = torchmetrics.F1Score(task='multilabel', num_labels=91)
        # preds = torch.sigmoid(preds); preds = (preds > 0.5).float()
        acc = accuracy(preds, gts.long())
    elif args.eval == 'multi-class':
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=91)
        preds = preds.argmax(dim=1); gts = gts.argmax(dim=1)
        acc = accuracy(preds, gts).item()
    print(f'The {args.eval} evaluation results is {acc}')

def novel(novel_loader, model):
    model.eval()
    scenario_name_decoder = scenario_name()
    with torch.no_grad():
        preds, gts = [], []
        for i, data in enumerate(tqdm(novel_loader)): 
            if i > 50:
                break
            context_audio_tags = audio_model.default_tagging(data['audio'].to('cuda'))
            # print(context_audio_tags)
            pred, gt = model(data, train=False, sequence=args.sequence)
            # pred = torch.sigmoid(pred) > 0.5
            # set the max value to 1
            pred = torch.zeros_like(pred).scatter(1, pred.argmax(dim=1).unsqueeze(1), 1)
            pred = scenario_name_decoder.decode(pred)
            gt = scenario_name_decoder.decode(gt)
            print(pred, gt)

class scenario_name():
    def __init__(self):
        import json
        self.scenario_map = json.load(open('resources/scenario_map.json', 'r'))
        # change the key of sceanrio_map from string to int
        self.scenario_map = {int(k):v for k,v in self.scenario_map.items()}
    def decode(self, scenario_vec):
        '''
        scenario_vec: torch.Tensor, [B, 91], multi-label
        '''
        scenario_name = []
        for i in range(scenario_vec.shape[0]):
            scenario_name.append([self.scenario_map[j] for j in range(91) if scenario_vec[i, j] == 1])
        return scenario_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--log', type=str, default=None) # work on existing log, auto make a new log by time, or work on a specific log
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ego4d_narration', choices=['ego4d_narration', 'ego4d_free', 'egoexo'])
    parser.add_argument('--novel_scenarios', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # split the dataset by novel scenario 
    parser.add_argument('--sequence', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=5) # if train = False, epoch is set to 0
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--window_sec', type=int, default=2)
    parser.add_argument('--mode', type=str, default='scenario', choices=['scenario', 'novel'])
    parser.add_argument('--eval', type=str, default='multi-label', choices=['multi-label', 'multi-class']) 
    args = parser.parse_args()
    model = Body_Sound(sequence=args.sequence).to('cuda')

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
    elif args.dataset == 'ego4d_free':
        dataset = Ego4D_Free(window_sec=args.window_sec, modal=['audio', 'imu']) # or you can set window_sec = sequence * 2 to make it fair
    elif args.dataset == 'egoexo':
        dataset = EgoExo_atomic(window_sec=args.window_sec, modal=['audio', 'imu'])
        dataset.prune_slience()
    support_idx, novel_idx = dataset.split_new_scenario(novel_scenario=args.novel_scenarios)
    support_dataset = torch.utils.data.Subset(dataset, support_idx)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    novel_dataset = torch.utils.data.Subset(dataset, novel_idx)
    novel_loader = torch.utils.data.DataLoader(novel_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    if args.train:
        Epoch = args.epoch
    else:
        Epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if args.mode == 'scenario':
        scenario(support_loader, support_loader, model, log_dir, args)
    elif args.mode == 'novel':
        audio_model = Mobilenet_Encoder('mn10_as').to('cuda')
        novel(novel_loader, model)
   