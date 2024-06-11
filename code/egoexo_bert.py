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
                                               collate_fn=Preprocess4Mask(json.load(open('mask.json', 'r'))))
    
    test_dataset = EgoExo_atomic(split='val')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_acc = 100, 100
    log_interval = 10
    test_interval = 500
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
                    torch.save(model.state_dict(), ckpt_dir + 'bert_{}_{}.pth'.format(i, test_loss))
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


if __name__ == "__main__":
    # Define the model
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import json
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/" + time_now
    ckpt_dir = "ckpts/" + time_now
    writer = SummaryWriter(log_dir=log_dir)

    device = 'cuda'
    model_cfg = json.load(open('limu_bert.json', 'r'))['base_v1']
    model = LIMUBertModel4Pretrain(model_cfg).to(device)
    main(model, device)
