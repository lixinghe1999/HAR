import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
class Capture24_Dataset(Dataset):
    def __init__(self, dataset_folder='capture24', supervised=False, split='train', seq_len=120, pipeline=[]):
        Xs, Ys = [], []
        num_users = 20
        for i in range(1, num_users+1):
            # x_name P001_X.npy, y_name P001_Y.npy
            x_name = 'P' + str(i).zfill(3) + '_X.npy'
            y_name = 'P' + str(i).zfill(3) + '_Y.npy'
            Xs.append(np.load(os.path.join(dataset_folder, x_name)))
            Ys.append(np.load(os.path.join(dataset_folder, y_name)))
        train_user = int(num_users * 0.8)
        if split == 'train':
            self.Xs = np.concatenate(Xs[:train_user])
            self.Ys = np.concatenate(Ys[:train_user])
        else:
            self.Xs = np.concatenate(Xs[train_user:])
            self.Ys = np.concatenate(Ys[train_user:])

        anno_label_dict = pd.read_csv(os.path.join('/home/lixing/har/dataset/capture24', 'annotation-label-dictionary.csv'), 
                                      index_col='annotation', dtype='string')
        # convert anno_label_dict['label:Willetts2018'] to dict
        self.Y_map = {k: v for k, v in zip(anno_label_dict.index, anno_label_dict['label:Willetts2018'])}
        # convert Y_map to unique index
        self.idx_map = {v: i for i, v in enumerate(np.unique(list(self.Y_map.values())))}
        

        self.supervised = supervised
        self.pipe_line = pipeline
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, idx):
        imu = self.Xs[idx, ::5] # downsample to 20Hz
        for pipe in self.pipe_line:
            imu = pipe(imu)
        if self.supervised:
            label = self.idx_map[self.Y_map[self.Ys[idx]]]
            return {'imu': imu, 'label': label}
        else:
            return {'imu': imu, 'label': None}