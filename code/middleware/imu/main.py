from bert import LIMUBertModel4Pretrain, ClassifierGRU, BERTClassifier
from limu_bert_dataset import Baseline_Dataset
# from capture24_dataset import Capture24_Dataset
from preprocess import Preprocess4Mask, Preprocess4Normalization, Preprocess4Rotation, Preprocess4Sample

import torch
from tqdm import tqdm
import torchmetrics
import json
import numpy as np

def main(model, device, num_epochs):
    batch_size = 32
    lr = 1e-5
    train_dataset = Baseline_Dataset(datasets=['uci'], split='train', supervised=False)
    test_dataset = Baseline_Dataset(datasets=['uci'], split='val', supervised=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                               collate_fn=Preprocess4Mask(json.load(open('config/mask.json', 'r'))))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_train_loss = 100
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
        if best_train_loss > train_loss:
            best_train_loss  = train_loss
            best_test_loss = test_loss
            model_best = model.state_dict()
    torch.save(model_best, log_dir + '/best.pth')
    print(f"Best Train Loss: {best_train_loss}, Best Test Loss: {best_test_loss}")
    writer.close()

def test(dataset, model, device):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, 
                                              collate_fn=Preprocess4Mask(json.load(open('config/mask.json', 'r'))))
    test_loss = 0
    with torch.no_grad():
        for i, (mask_seqs, masked_pos, seqs) in enumerate(test_loader):
            seq_recon = model(mask_seqs.to(device), masked_pos.to(device))
            loss = torch.nn.functional.l1_loss(seq_recon, seqs.to(device))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    return test_loss
def fine_tune(model, device, num_epochs):
    batch_size = 32
    lr = 1e-5
    pipeline = [Preprocess4Normalization(6), Preprocess4Rotation()]
    pipeline = []
    train_dataset = Baseline_Dataset(datasets=['uci'], split='train', supervised=True, seq_len=seq_len, pipeline=pipeline)
    test_dataset = Baseline_Dataset(datasets=['uci'], split='val', supervised=True, seq_len=seq_len, pipeline=pipeline)

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
        print(f" Epoch: {E}, Train Loss: {train_loss}, Test Accuracy: {test_loss}")
        if best_loss > train_loss:
            best_loss = train_loss
            best_acc = test_loss
            model_best = model.state_dict()
    torch.save(model_best, './{}_{}.pth'.format(round(test_loss.item(),3), seq_len))
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

class LIMU_BERT_Inferencer():
    def __init__(self, model_cfg, classifier_cfg, ckpt, device, class_name = ["walking", "upstairs", "downstairs", "sitting", "standing", "lying"]):
        model_cfg = json.load(open(model_cfg, 'r'))['base_v1']
        classifier_cfg = json.load(open(classifier_cfg, 'r'))['gru_v1']
        self.model = BERTClassifier(model_cfg, classifier = ClassifierGRU(classifier_cfg, input=72, output=len(class_name))).to(device)
        self.model.load_state_dict(torch.load(ckpt), strict=False)
        self.device = device
        self.class_name = class_name
        self.seq_len = model_cfg['seq_len']
        self.class_name = class_name
        print('LIMU BERT Inferencer is ready', self.seq_len)
    def infer(self, imu, sr=800):
        '''
        imu: [N, 6]
        '''
        if type(imu) == np.ndarray:
            imu = torch.from_numpy(imu)       
        down_sample_rate = sr // 20
        imu = imu[::down_sample_rate]
        if imu.shape[0] % self.seq_len != 0:
            if imu.shape[0] < self.seq_len: # pad to be the same length
                imu = torch.nn.functional.pad(imu, (0, 0, 0, self.seq_len - imu.shape[0]))
            else:# pad to be the multiple of seq_len
                pad_len = self.seq_len - imu.shape[0] % self.seq_len
                imu = torch.nn.functional.pad(imu, (0, 0, 0, pad_len))
        imu = imu.reshape(-1, self.seq_len, 6)
        with torch.no_grad():
            representations = self.model(imu.to(device=self.device, dtype=torch.float32))
        idxs = torch.argmax(representations, dim=1).cpu().numpy()
        return idxs
if __name__ == "__main__":
    # Define the model
    device = 'cuda'
    seq_len = 200 # control the length of input
    model_cfg = json.load(open('config/limu_bert_{}.json'.format(seq_len), 'r'))['base_v1']
    classifier_cfg = json.load(open('config/classifier.json', 'r'))['gru_v1']
    pretrain = False
    ckpt = None
    dataset = 'uci'
    dataset = 'capture24'
    classes_map = {'uci': 6, 'capture24': 6}
    num_classes = classes_map[dataset]
    pretrain_epochs = 200
    fine_tune_epochs = 50

    if not pretrain:
        model = BERTClassifier(model_cfg, classifier = ClassifierGRU(classifier_cfg, input=72, output=num_classes)).to(device)
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt), strict=False)
        fine_tune(model, device, fine_tune_epochs)
    else:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "resources/" + time_now
        writer = SummaryWriter(log_dir=log_dir)
        model = LIMUBertModel4Pretrain(model_cfg).to(device)
        main(model, device, pretrain_epochs)
