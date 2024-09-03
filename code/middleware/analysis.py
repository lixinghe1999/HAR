import sys
sys.path.append('imu')
from imu.limu_bert_dataset import Baseline_Dataset
from imu.main import LIMU_BERT_Inferencer
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# def trivial_middleware(imu):
#     imu_mean = np.mean(imu, axis=0)
#     imu_std = np.std(imu, axis=0)
#     if imu_mean[0] < (imu_mean[1] + imu_mean[2]) / 2: # very special case
#         return "lying"
#     else:
#         if np.mean(imu_std[3:]) > 0.1: # moving - walk, up or down
            
#         else: # not moving - sit, stand


test_dataset = Baseline_Dataset(datasets=['uci'], split='val', supervised=True, seq_len=20)
imu_middleware = LIMU_BERT_Inferencer(model_cfg='imu/config/limu_bert_20.json', classifier_cfg='imu/config/classifier.json', ckpt='imu/0.872_20.pth', device='cuda')

# acc_num = 0
# for i, data in enumerate(test_dataset):
#     imu_output = imu_middleware.infer(data['imu'], sr=20)
#     if imu_output == data['label']:
#         acc_num += 1
# print(f"Accuracy: {acc_num / len(test_dataset)}")
# accuracy is right

# cluster the data by label
label_cluster = {}
for data in test_dataset:
    label = data['label']
    if label not in label_cluster:
        label_cluster[label] = []
    label_cluster[label].append(data['imu'])

class_names = ["walking", "upstairs", "downstairs", "sitting", "standing", "lying"]
num_examples = 5
fig, axs = plt.subplots(len(label_cluster), 2, figsize=(10, 5))
features = []
for i, label in enumerate(label_cluster):
    label_name = class_names[label]
    label_cluster[label] = np.stack(label_cluster[label])
    print(label_cluster[label].shape)
    data_mean = np.mean(label_cluster[label], axis=(0, 1))
    data_std = np.std(label_cluster[label], axis=(0, 1))
    
    # plot mean and upper and bottom bound
    axs[i, 0].plot(data_mean[:3])
    axs[i, 0].fill_between(range(3), data_mean[:3] - data_std[:3], data_mean[:3] + data_std[:3], alpha=0.5)
    axs[i, 1].plot(data_mean[3:])
    axs[i, 1].fill_between(range(3), data_mean[3:] - data_std[3:], data_mean[3:] + data_std[3:], alpha=0.5)
    # show the label name
    axs[i, 0].set_title(label_name)

    feature1 = data_mean[0] > (data_mean[1] + data_mean[2])/2 # distinguish lying and other
    feature2 = np.mean(data_std[3:]) # distinguish moving and not moving
    feature3 = np.diff(data_mean[3:]).mean() # distinguish walking, upstairs, downstairs
    feature4 = np.diff(data_mean[:3]).mean() # distinguish sitting, standing
    features += [feature1, feature2, feature3, feature4]
plt.savefig('mean.png')

fig, axs = plt.subplots(4, 1, figsize=(10, 5))
features = np.array(features).reshape(-1, 4)
print(features.shape)
for i in range(4):
    axs[i].plot(features[:, i])
plt.savefig('hist.png')

    

    # example_idx = np.random.choice(range(len(label_cluster[label])), num_examples)
    # example_recording = label_cluster[label][example_idx]
    # fig, axs = plt.subplots(num_examples, 1, figsize=(10, 5))
    # for j, recording in enumerate(example_recording):
    #     axs[j].plot(recording)
    # plt.savefig('{}.png'.format(label_name))
