from models.imu_models import  MW2StackRNNPooling, TransformerEncoder
from models.classification_head import Head
from models.text_models import BERT
from models.loss import InfoNCE
from ego_dataset import Ego4D_Moment, Ego4D_Narration, IMU2CLIP_Dataset
import torch
from tqdm import tqdm

def discrimate_train(train_dataset, test_dataset, model, optimizer, device, num_epochs=10):
    print("Start training with weights of ", weights)
    metric = torch.nn.CrossEntropyLoss(weight=None,
                                       # torch.from_numpy(weights).to(device)
                                       )
    best_loss = 100
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataset)
        train_loss = 0
        for i, data in enumerate(pbar):
            target = data['label'].to(device)
            imu_output = model['imu'](data['imu'].to(device))
            loss = metric(imu_output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_description(f"Epoch {epoch}, Loss: {train_loss/(i+1)}")
        train_loss /= len(train_dataset)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, data in enumerate(test_dataset):
                target = data['label'].to(device)
                imu_output = model['imu'](data['imu'].to(device))
                logits = torch.argmax(imu_output, dim=1)
                y_true.extend(list(target.cpu().numpy()))
                y_pred.extend(list(logits.cpu().numpy()))
        from sklearn.metrics import confusion_matrix, accuracy_score
        matrix = confusion_matrix(y_true, y_pred)
        acc_all = matrix.diagonal()/matrix.sum(axis=1)
        acc = acc_all.mean()
        print(f"Epoch {epoch}, Accuracy: {acc_all}, balanced accuracy: {acc}, accuracy: {accuracy_score(y_true, y_pred)}")
        if best_loss > train_loss:
            best_loss = train_loss
            best_acc = acc
            import matplotlib.pyplot as plt
            plt.imshow(matrix)
            plt.axis('off')
            plt.colorbar()
            plt.savefig('confusion_matrix.png')
            torch.save(model['imu'].state_dict(), 'best_model.pth')
    print(f"Best Loss: {best_loss}, Best Accuracy: {best_acc}")
    return




if __name__ == "__main__":
    # Define the model
    device = 'cuda'
    imu_model = MW2StackRNNPooling(size_embeddings=384)
    imu_model = Head(imu_model, size_embeddings=384, n_classes=4)
    imu_model = imu_model.to(device)
    lr = 4e-5
    optimizer = torch.optim.Adam(imu_model.parameters(), lr=lr)

    model = {
        # 'text': BERT(), 
        'imu': imu_model}
    dataset = IMU2CLIP_Dataset(window_sec=2.5, modality=['imu',], split='train')
    weights = dataset.get_weight()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    dataset = IMU2CLIP_Dataset(window_sec=2.5, modality=['imu',], split='val')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    # contrastive_train(train_loader, model, optimizer, device)
    discrimate_train(train_loader, test_loader, model, optimizer, device)