import os
import numpy as np
from torch.utils.data import Dataset
Dataset_folder = '/home/lixing/har/dataset'
class Baseline_Dataset(Dataset):
    '''
    load npy dataset from the folder 'small_dataset'
    [hhar, motion, shoaib, uci]
    '''
    def __init__(self, datasets=['hhar', 'motion', 'shoaib', 'uci'], supervised=False, split='train', seq_len=120, pipeline=[]):
        datas, labels = [], []
        if supervised: # supervised learning, only use one dataset since the label is different
            assert len(datasets) == 1
        version = '20_{}'.format(seq_len)
        for data_dir in datasets:
            data_dir = os.path.join(Dataset_folder, data_dir)
            data = np.load(data_dir + f'/data_{version}.npy').astype(np.float32)
            arr = np.arange(data.shape[0])
            np.random.seed(0)
            np.random.shuffle(arr)
            data = data[arr]
            if data.shape[2] > 6:
                data = data[:, :, :6]
            if split == 'train':
                data = data[:int(0.8 * data.shape[0])]
            else:
                data = data[int(0.8 * data.shape[0]):]
            datas.append(data)

            label = np.load(data_dir + f'/label_{version}.npy').astype(np.int64)
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
        self.pipe_line = pipeline
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        imu = self.data[idx]
        # print(imu.shape, imu.dtype)
        for pipe in self.pipe_line:
            imu = pipe(imu)
        # print(imu.shape, imu.dtype)
        if self.supervised:
            return {'imu': imu, 'label': self.labels[idx]}
        else:
            return {'imu': imu, 'label': None}