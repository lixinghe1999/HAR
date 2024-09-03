'''
Copy from UniHAR paper: https://dapowan.github.io/wands_unihar/
'''

import numpy as np
import torch

from scipy import signal, interpolate
from scipy.stats import special_ortho_group

def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)
class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg['mask_ratio'] # masking probability
        self.mask_alpha = mask_cfg['mask_alpha']
        self.max_gram = mask_cfg['max_gram']
        self.mask_prob = mask_cfg['mask_prob']
        self.replace_prob = mask_cfg['replace_prob']

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data
    def __call__(self, instance):
        instances, mask_pos_indexs, seqs = [], [], []
        for instance_ in instance:
            instance_mask, mask_pos_index, seq = self.__call__single(instance_)
            instances.append(instance_mask)
            mask_pos_indexs.append(mask_pos_index)
            seqs.append(seq)
        instances = np.array(instances)
        mask_pos_indexs = np.array(mask_pos_indexs)
        seqs = np.array(seqs)
        # instances = np.zeros((16, 320, 12), dtype=np.float32)
        # mask_pos_indexs = np.zeros((16, 15), dtype=np.int64)
        # seqs = np.zeros((16, 15, 12), dtype=np.float32)
        return torch.tensor(instances), torch.tensor(mask_pos_indexs), torch.tensor(seqs)
    def __call__single(self, instance):
        instance = instance['imu']
        shape = instance.shape
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)
    

class Pipeline:
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):

    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new


class Preprocess4Sample(Pipeline):

    def __init__(self, seq_len, temporal=0.4, temporal_range=[0.8, 1.2]):
        super().__init__()
        self.seq_len = seq_len
        self.temporal = temporal
        self.temporal_range = temporal_range

    def __call__(self, instance):
        if instance.shape[0] == self.seq_len:
            return instance
        if self.temporal > 0:
            temporal_prob = np.random.random()
            if temporal_prob < self.temporal:
                x = np.arange(instance.shape[0])
                ratio_random = np.random.random() * (self.temporal_range[1] - self.temporal_range[0]) + self.temporal_range[0]
                seq_len_scale = int(np.round(ratio_random * self.seq_len))
                index_rand = np.random.randint(0, high=instance.shape[0] - seq_len_scale)
                instance_new = np.zeros((self.seq_len, instance.shape[1]))
                for i in range(instance.shape[1]):
                    f = interpolate.interp1d(x, instance[:, i], kind='linear')
                    x_new = index_rand + np.linspace(0, seq_len_scale, self.seq_len)
                    instance_new[:, i] = f(x_new)
                return instance_new
        index_rand = np.random.randint(0, high=instance.shape[0] - self.seq_len)
        return instance[index_rand:index_rand + self.seq_len, :]


class Preprocess4Noise(Pipeline):

    def __init__(self, p=1.0, mu=0.0, var=0.1):
        super().__init__()
        self.p = p
        self.mu = mu
        self.var = var

    def __call__(self, instance):
        if np.random.random() < self.p:
            instance += np.random.normal(self.mu, self.var, instance.shape)
        return instance


class Preprocess4Rotation(Pipeline):

    def __init__(self, sensor_dimen=3):
        super().__init__()
        self.sensor_dimen = sensor_dimen

    def __call__(self, instance):
        return self.rotate_random(instance)

    def rotate_random(self, instance):
        instance_new = instance.reshape(instance.shape[0], instance.shape[1] // self.sensor_dimen, self.sensor_dimen)
        rotation_matrix = special_ortho_group.rvs(self.sensor_dimen)
        for i in range(instance_new.shape[1]):
            instance_new[:, i, :] = np.dot(instance_new[:, i, :], rotation_matrix)
        return instance_new.reshape(instance.shape[0], instance.shape[1])


class Preprocess4Permute(Pipeline):

    def __init__(self, segment_size=4):
        super().__init__()
        self.segment_size = segment_size

    def __call__(self, instance):
        original_shape = instance.shape
        instance = instance.reshape(self.segment_size, instance.shape[0]//self.segment_size, -1)
        order = np.random.permutation(self.segment_size)
        instance = instance[order, :, :]
        return instance.reshape(original_shape)


class Preprocess4Flip(Pipeline):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, instance):
        if np.random.random() < self.p:
            instance = instance[::-1, :].copy()
        return instance


class Preprocess4STFT(Pipeline):

    def __init__(self, window=50, cut_off_frequency=17, fs=20):
        super().__init__()
        self.window = window
        self.cut_off_frequency = cut_off_frequency
        self.fs = fs

    def __call__(self, instance):
        instance_new = []
        for i in range(instance.shape[1]):
            f, t, Zxx = signal.stft(instance[:, i], self.fs, nperseg=self.window)
            instance_new.append(Zxx[:self.cut_off_frequency, :])
        instance_new = np.abs(np.vstack(instance_new).transpose([1, 0]))
        return instance_new