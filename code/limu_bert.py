from models.bert_models import LIMUBertModel4Pretrain, ClassifierGRU, BERTClassifier
from egoexo_dataset import Baseline_Dataset
import torch
from tqdm import tqdm
from utils.mask import Preprocess4Mask
import torchmetrics
import json

def main(model, device, num_epochs):
    batch_size = 32
    lr = 1e-5
    train_dataset = Baseline_Dataset(datasets=['uci'], split='train', supervised=False)
    test_dataset = Baseline_Dataset(datasets=['uci'], split='val', supervised=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                               collate_fn=Preprocess4Mask(json.load(open('models/mask.json', 'r'))))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_acc = 100, 100
    for E in range(num_epochs):
        train_loss = 0
        pbar = tqdm(train_loader)
        for i, (mask_seqs, masked_pos, seqs) in enumerate(pbar):
            optimizer.zero_grad()
            seq_recon = model(mask_seqs.to(device), masked_pos.to(device))
            loss = torch.nn.functional.mse_loss(seq_recon, seqs.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item()}")
        train_loss /= len(train_loader)
        test_loss = test(test_dataset, model, device)
        print(f" Epoch: {E}, Train Loss: {train_loss}, Test Loss: {test_loss}")
        writer.add_scalar('Loss/train', train_loss, E)
        writer.add_scalar('Loss/test', test_loss, E)
        if best_loss > train_loss:
            best_loss  = train_loss
            best_acc = test_loss
            model_best = model.state_dict()
    torch.save(model_best, log_dir + '/best.pth'.format(test_loss))
    print(f"Best Loss: {best_loss}, Best Accuracy: {best_acc}")
    writer.close()

def test(dataset, model, device):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, 
                                              collate_fn=Preprocess4Mask(json.load(open('models/mask.json', 'r'))))
    test_loss = 0
    with torch.no_grad():
        for i, (mask_seqs, masked_pos, seqs) in enumerate(test_loader):
            seq_recon = model(mask_seqs.to(device), masked_pos.to(device))
            loss = torch.nn.functional.l1_loss(seq_recon, seqs.to(device))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    return test_loss
def fine_tune(model, device, num_epochs):
    batch_size = 4
    lr = 1e-5
    train_dataset = Baseline_Dataset(datasets=['uci'], split='train', supervised=True)
    test_dataset = Baseline_Dataset(datasets=['uci'], split='val', supervised=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_acc = 100, 100
    for E in range(num_epochs):
        train_loss = 0
        pbar = tqdm(train_loader)
        for i, dict_out in enumerate(pbar):
            optimizer.zero_grad()
            cls_pred = model(dict_out['imu'].to(device))
            loss = torch.nn.functional.cross_entropy(cls_pred, dict_out['label'].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item()}")
        train_loss /= len(train_loader)
        test_loss = test_supervised(test_dataset, model, device)
        print(f" Epoch: {E}, Train Loss: {train_loss}, Test Loss: {test_loss}")
        if best_loss > train_loss:
            best_loss = train_loss
            best_acc = test_loss
            model_best = model.state_dict()
    torch.save(model_best, './{}_finetune.pth'.format(round(test_loss.item(),2)))
    print(f"Best Loss: {best_loss}, Best Accuracy: {best_acc}")
    return model_best

def test_supervised(dataset, model, device):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False,)
    pred, gt = [], []
    with torch.no_grad():
        for i, dict_out in enumerate(test_loader):
            cls_pred = model(dict_out['imu'].to(device))
            pred.append(cls_pred)
            gt.append(dict_out['label']) 
    pred, gt = torch.cat(pred, dim=0).cpu(), torch.cat(gt, dim=0)
    test_loss = torchmetrics.functional.accuracy(pred, gt, task='multiclass', num_classes=num_classes)
    return test_loss

def save_embedding(data_dir, model, device, hop_frame=800):
    import os
    import numpy as np
    takes = os.listdir(data_dir)
    with torch.no_grad():
        for take in tqdm(takes):
            take_path = os.path.join(data_dir, take)    
            files = os.listdir(take_path)
            imu_files = [f for f in files if f.endswith('_noimagestreams.npy')]
            if len(imu_files) == 0:
                continue
            imu = np.load(os.path.join(take_path, imu_files[0])).astype(np.float32)
            n_windows = int(np.ceil(imu.shape[1] / hop_frame))
            save_data = []
            for i in range(n_windows):
                start = i * hop_frame
                end = min((i+1) * hop_frame, imu.shape[1])
                data = imu[:, start:end]
                if data.shape[1] < hop_frame:
                    data = np.pad(data, ((0, 0), (0, int(hop_frame - data.shape[1]))), 'constant', constant_values=0)
                data = data.T[np.newaxis, ::5, :]
                representations = model(torch.from_numpy(data).to(device))
                save_data.append(representations)
            save_data = torch.cat(save_data, dim=0)
            save_data = save_data.cpu().numpy()
            np.save(os.path.join(take_path, 'limu_bert.npy'), save_data)

class LIMU_BERT_Inferencer():
    def __init__(self, ckpt, device, class_name = ["walking", "upstairs", "downstairs", "sitting", "standing", "lying"]):
        model_cfg = json.load(open('models/limu_bert.json', 'r'))['base_v1']
        classifier_cfg = json.load(open('models/classifier.json', 'r'))['gru_v1']
        self.model = BERTClassifier(model_cfg, classifier = ClassifierGRU(classifier_cfg, input=72, output=len(class_name))).to(device)
        self.model.load_state_dict(torch.load(ckpt), strict=False)
        self.device = device
        self.class_name = class_name
        print('warning, the unit for accelerometer should be m/sec^2')
    def infer(self, imu, sr=800):
        down_sample_rate = sr // 20
        imu = imu[::down_sample_rate]
        if imu.shape[-1] == 9:
            imu = imu[None, :, :6]
        elif imu.shape[-1] == 12:
            imu = imu.reshape(2, -1, 6)
        else:
            imu = imu[None, :, :]
        imu = torch.from_numpy(imu).float()
        with torch.no_grad():
            representations = self.model(imu.to(self.device))
        idxs = torch.argmax(representations, dim=1)
        return [self.class_name[idx] for idx in idxs]
if __name__ == "__main__":
    # Define the model
    device = 'cuda'
    model_cfg = json.load(open('models/limu_bert.json', 'r'))['base_v1']
    classifier_cfg = json.load(open('models/classifier.json', 'r'))['gru_v1']
    ckpt = None
    ckpt = 'logs/20240621-104842/best.pth'
    # ckpt = './limu_v1.pt'
    dataset = 'uci'
    classes_map = {'uci': 6}
    num_classes = classes_map[dataset]
    pretrain_epochs = 200
    fine_tune_epochs = 50

    if ckpt is not None:
        model = BERTClassifier(model_cfg, classifier = ClassifierGRU(classifier_cfg, input=72, output=num_classes)).to(device)
        model.load_state_dict(torch.load(ckpt), strict=False)
        fine_tune(model, device, fine_tune_epochs)
        # save_embedding('../dataset/egoexo/takes/', model, device)
    else:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/" + time_now
        writer = SummaryWriter(log_dir=log_dir)
        model = LIMUBertModel4Pretrain(model_cfg).to(device)
        main(model, device, pretrain_epochs)
