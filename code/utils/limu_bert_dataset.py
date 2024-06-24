import os
import numpy as np
from torch.utils.data import Dataset
class Baseline_Dataset(Dataset):
    '''
    load npy dataset from the folder 'small_dataset'
    [hhar, motion, shoaib, uci]
    '''
    def __init__(self, datasets=['hhar', 'motion', 'shoaib', 'uci'], supervised=False, split='train'):
        datas, labels = [], []
        if supervised:
            assert len(datasets) == 1
        for data_dir in datasets:
            data_dir = os.path.join('small_dataset', data_dir)
            data = np.load(data_dir + '/data_20_120.npy').astype(np.float32)
            arr = np.arange(data.shape[0])
            np.random.shuffle(arr)
            data = data[arr]
            if data.shape[2] > 6:
                data = data[:, :, :6]
            if split == 'train':
                data = data[:int(0.8 * data.shape[0])]
            else:
                data = data[int(0.8 * data.shape[0]):]
            datas.append(data)

            label = np.load(data_dir + '/label_20_120.npy').astype(np.int64)
            label = label[:, 0, 0]
            label = label[arr]
            if split == 'train':
                label = label[:int(0.8 * label.shape[0])]
            else:
                label = label[int(0.8 * label.shape[0]):]
            assert data.shape[0] == label.shape[0]            
            labels.append(label)

        self.data = np.concatenate(datas, axis=0)
        if supervised:
            self.labels = np.concatenate(labels, axis=0)
        self.supervised = supervised
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        if self.supervised:
            return {'imu': self.data[idx], 'label': self.labels[idx]}
        else:
            return {'imu': self.data[idx], 'label': None}