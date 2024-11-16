from models.multi_modal import Multi_modal_model
import torch
import time
if __name__ == '__main__':
    device = 'cpu'
    model = Multi_modal_model().to(device)
    data = {
        'audio': torch.randn(10, 16000).to(device),
        'imu': torch.randn(10, 6, 200).to(device),
        'scenario': torch.randint(0, 91, (10,)).to(device)
    }

    for i in range(100):
        if i == 10:
            start = time.time()
        loss = model(data, device=device)
    end = time.time()
    print('latency:', (end - start) / 90)