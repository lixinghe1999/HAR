from models.bert_models import LIMUBertModel4Pretrain
from egoexo_dataset import EgoExo_atomic
import torch
from tqdm import tqdm
from utils.mask import Preprocess4Mask
import time
def main(model, device, num_epochs=1):
    batch_size = 64
    lr = 1e-5

    train_dataset = EgoExo_atomic(split='train',)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                               collate_fn=Preprocess4Mask(json.load(open('models/mask.json', 'r'))))
    
    test_dataset = EgoExo_atomic(split='val')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_acc = 100, 100
    log_interval = 10
    test_interval = 1000
    for _ in range(num_epochs):
        train_loss = 0
        for i, (mask_seqs, masked_pos, seqs) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            seq_recon = model(mask_seqs.to(device), masked_pos.to(device))
            loss = torch.nn.functional.mse_loss(seq_recon, seqs.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % log_interval == 0:
                writer.add_scalar('Train_loss/train', loss.item(), i)
            if i % test_interval == 0 and i > 0:
                train_loss /= len(train_loader)
                test_loss = test(test_dataset, model, device, i)
                if best_loss > train_loss:
                    best_loss = train_loss
                    best_acc = test_loss
                    torch.save(model.state_dict(), log_dir + '/bert_{}_{}.pth'.format(i, test_loss))
    print(f"Best Loss: {best_loss}, Best Accuracy: {best_acc}")
    writer.close()

def test(dataset, model, device, idx):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, 
                                              collate_fn=Preprocess4Mask(json.load(open('mask.json', 'r'))))
    test_loss = 0
    with torch.no_grad():
        for i, (mask_seqs, masked_pos, seqs) in enumerate(test_loader):
            seq_recon = model(mask_seqs.to(device), masked_pos.to(device))
            loss = torch.nn.functional.l1_loss(seq_recon, seqs.to(device))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    writer.add_scalar('Test_Error/mean', test_loss, idx)
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


if __name__ == "__main__":
    # Define the model
    import json
    device = 'cuda'
    model_cfg = json.load(open('models/limu_bert.json', 'r'))['base_v1']
    ckpt = 'logs/20240613-121500/bert_4000_1.192920057806167.pth'
    # ckpt = None
    if ckpt is not None:
        model = LIMUBertModel4Pretrain(model_cfg, output_embed=True).to(device)
        model.load_state_dict(torch.load(ckpt))
        save_embedding('../dataset/egoexo/takes/', model, device)
    else:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/" + time_now
        writer = SummaryWriter(log_dir=log_dir)
        model = LIMUBertModel4Pretrain(model_cfg, output_embed=False).to(device)
        main(model, device)
