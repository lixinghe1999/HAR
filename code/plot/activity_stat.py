import numpy as np
import torchmetrics
import torch

def class_wise_acc(output_dir = 'resources/activity_audio/'):
    
    gts = np.load(output_dir + 'gts.npy')
    preds = np.load(output_dir + 'preds.npy')
    print(gts.shape, preds.shape)

    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=50, average=None)
    acc = accuracy(torch.tensor(preds), torch.tensor(gts))
    return acc

acc_audio = class_wise_acc('resources/activity_audio/')
acc_imu = class_wise_acc('resources/activity_imu/') 

acc_audio *= 100
acc_imu *= 100
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2)
axs[0].bar(range(50), acc_audio, label='audio')
axs[0].bar(range(50), acc_imu, label='imu')
axs[0].legend()
axs[0].set_title('Activity classwise accuracy')
axs[0].set_ylabel('Accuracy (%)')   

modality_gap = acc_audio - acc_imu
axs[1].bar(range(50), modality_gap)
axs[1].set_title('Modality gap, Audio - IMU')

plt.savefig('figs/activity_classwise.png')
