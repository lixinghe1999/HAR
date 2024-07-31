import sys
sys.path.append('..')

from models.imu_models import TransformerEncoder
from models.audio_models import AudioTagging
from models.classification_head import Head
from ego4d.ego4d_dataset import Ego4D_Moment, Ego4D_Narration, IMU2CLIP_Dataset
import torch
from tqdm import tqdm
import psutil

def get_cpu_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1e6  # in MB

def discrimate_train(train_dataset, test_dataset, model, optimizer, device, num_epochs=10):
    metric = torch.nn.CrossEntropyLoss()
    # metric = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    best_loss = 100
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataset)
        train_loss = 0
        for i, data in enumerate(pbar):
            target = data['label'].to(device)
            imu_output = model(data['imu'].to(device))
            loss = metric(imu_output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            pbar.set_description(f"Epoch {epoch}, Loss: {train_loss/(i+1)}")
            # if i % 10 == 0:
            #     print(f"Memory Usage: {get_cpu_memory_usage()}")
        train_loss /= len(train_dataset)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, data in enumerate(test_dataset):
                target = data['label'].to(device)
                imu_output = model(data['imu'].to(device))
                logits = torch.argmax(imu_output, dim=1)
                y_true.extend(list(target.cpu().numpy()))
                y_pred.extend(list(logits.cpu().numpy()))
        from sklearn.metrics import confusion_matrix, accuracy_score
        matrix = confusion_matrix(y_true, y_pred)
        acc_all = matrix.diagonal()/(matrix.sum(axis=1)+0.001)
        acc = acc_all.mean()
        print(f"Epoch {epoch}, balanced accuracy: {acc}, accuracy: {accuracy_score(y_true, y_pred)}")
        if best_loss > train_loss:
            best_loss = train_loss
            best_acc = acc
            best_matrix = matrix
    import matplotlib.pyplot as plt
    plt.imshow(best_matrix)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('confusion_matrix.pdf')
    plt.cla()
    print(f"Best Loss: {best_loss}, Best Accuracy: {best_acc}")
    return




if __name__ == "__main__":

    # Define the model
    train_dataset = Ego4D_Moment(window_sec=2.5, modality=['imu', 'cluster_label'], split='train')
    test_dataset = Ego4D_Moment(window_sec=2.5, modality=['imu', 'cluster_label'], split='val')
    train_dataset.align_labels(train_dataset.labels, test_dataset.labels)
    test_dataset.align_labels(train_dataset.labels, test_dataset.labels)
    # dataset = IMU2CLIP_Dataset(window_sec=2.5, modality=['imu'], split='train')
    weights = train_dataset.weights
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    
    # dataset = IMU2CLIP_Dataset(window_sec=2.5, modality=['imu'], split='val')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    device = 'cuda'
   
    imu_model = TransformerEncoder(size_embeddings=384)
    imu_model = Head(imu_model, size_embeddings=384, n_classes=train_dataset.num_class)
    imu_model = imu_model.to(device)
    model = imu_model

    lr = 4e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    discrimate_train(train_loader, test_loader, model, optimizer, device)