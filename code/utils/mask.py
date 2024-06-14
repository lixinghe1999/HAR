import numpy as np
import torch
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