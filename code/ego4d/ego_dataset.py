import numpy as np
from torch.utils.data import Dataset
import os
from .extract_imu import get_ego4d_metadata
from .ego4d_utils import index_narrations, index_moments
from tqdm import tqdm
import math
import string
import librosa
import json
from tqdm import tqdm

from .text_cluster import close_to, cluster_plot, cluster_map

def clean_moment_text(narration_text: str) -> list:
    return (
        narration_text.replace("_", " ")
        .strip()
        .strip(string.punctuation)
        .lower()[:128]
    )
def clean_narration_text(narration_text: str) -> str:
    return (
        narration_text.replace("#C C ", "")
        .replace("#C", "")
        .replace("#unsure", "something")
        .strip()
        .strip(string.punctuation)
        .lower()[:128]
    )

def prepare_narration(narration_dict, metadata, meta_audio, meta_imu, modality, window_sec):
    window_idx = []
    for video_uid, narrations in (narration_dict.items()):
        duration = metadata[video_uid]["video_metadata"]["video_duration_sec"]
        if 'imu' in modality and metadata[video_uid]["has_imu"]:
            duration = min(duration, meta_imu[video_uid])
        for (timestamp, text, a_uid, _) in narrations:
            if not "#c" in text.lower(): #only the wearer
                continue
            if timestamp > duration or window_sec * 2 > duration:
                continue # skip if the timestamp is larger than the duration
            else:
                # check if it's the timestamp is at the very beginning
                if timestamp <= window_sec * 2:
                    w_s = 0.0
                    w_e = window_sec * 2
                # check if it's the time stamp is at the very end
                elif timestamp + window_sec * 2 >= duration:
                    w_s = duration - window_sec * 2
                    w_e = duration
                # else get a window of data around the timestamps
                else:
                    w_s = timestamp - window_sec
                    w_e = timestamp + window_sec
            w_s = int(math.floor(w_s))
            w_e = int(math.floor(w_e))
            try:
                assert w_e - w_s == window_sec * 2
            except AssertionError:
                continue

            input_dict = {
                "window_start": w_s,
                "window_end": w_e,
                "video_uid": video_uid,
                # "narration_uid": a_uid,
                "text": clean_narration_text(text),
            }
            window_idx.append(input_dict)
    return window_idx
def prepare_moment(moment_dict, metadata, meta_imu, window_sec):
    window_idx = []
    for video_uid, moments in tqdm(moment_dict.items()):
        if not metadata[video_uid]["has_imu"]:
            continue
        video_duration = metadata[video_uid]["video_metadata"]["video_duration_sec"]
        imu_duration = meta_imu[video_uid]
        duration = min(video_duration, imu_duration) # imu duration may be smaller the video duration
        # print(video_uid, video_duration, imu_duration)
        for (label, start_time, end_time) in moments:
            timestamp = (start_time + end_time) / 2
            if timestamp > duration:
                continue # skip if the timestamp is larger than the duration
            else:
                if timestamp <= window_sec * 2:
                    w_s = 0.0
                    w_e = window_sec * 2
                # check if it's the time stamp is at the very end
                elif timestamp + window_sec * 2 >= duration:
                    w_s = duration - window_sec * 2
                    w_e = duration
                # else get a window of data around the timestamps
                else:
                    w_s = timestamp - window_sec
                    w_e = timestamp + window_sec
            w_s = int(math.floor(w_s))
            w_e = int(math.floor(w_e))
            try:
                assert w_e - w_s == window_sec * 2
            except AssertionError:
                continue
            label = clean_moment_text(label)
            input_dict = {
                "window_start": w_s,
                "window_end": w_e,
                "video_uid": video_uid,
                "text": label,
            }
            window_idx.append(input_dict)
    print(f"There are {len(window_idx)} windows to process.")
    return window_idx
class Ego4D_Narration(Dataset):
    def __init__(self, pre_compute_json=None, folder='../dataset/ego4d/v2/', window_sec = 2, modal=['imu', 'audio']):
        self.folder = folder
        self.modal = modal
        self.metadata = get_ego4d_metadata('../dataset/ego4d/ego4d.json', "video")
        if pre_compute_json is not None:
            self.pre_compute_json = pre_compute_json
            with open(pre_compute_json, 'r') as f:
                self.window_idx = json.load(f)
        else:
            self.meta_imu = json.load(open('../dataset/ego4d/v2/annotations/meta_imu.json', 'r'))
            self.meta_audio = [v[:-4] for v in os.listdir('../dataset/ego4d/v2/audio')]
            filter_video_uid = []
            for video_uid in list(self.metadata.keys()):
                keep_or_not = False
                if "imu" in modal:
                    if self.metadata[video_uid]["has_imu"] and video_uid in self.meta_imu:
                        keep_or_not = True
                if "audio" in modal:
                    if video_uid in self.meta_audio:
                        keep_or_not = True
                if keep_or_not:
                    filter_video_uid.append(video_uid)
            narration_dict = index_narrations(filter_video_uid)
            self.window_idx = prepare_narration(narration_dict, self.metadata, self.meta_audio, self.meta_imu, self.modal, window_sec)
        print(f"There are {len(self.window_idx)} windows to process.")

        self.sr_imu = 200
        self.sr_audio = 16000
    def __len__(self):
        return len(self.window_idx)
    def process_item(self, item):
        data = self.__getitem__(item)
        RMS = float(np.sqrt(np.mean(data['audio']**2)))
        return RMS
    def cal_rms(self):
        # import matplotlib.pyplot as plt
        r = []
        for i in tqdm(range(self.__len__())):
            data = self.__getitem__(i)
            RMS = float(np.sqrt(np.mean(data['audio']**2)))
            self.window_idx[i]['rms'] = RMS
            r.append(RMS)
        # plt.hist(r, bins=100)
        # plt.savefig('figs/rms.png')
        self.save_json('resources/ego4d_rms.json')
    def save_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.window_idx, f, indent=4)    
    def __getitem__(self, i):
        dict_out = self.window_idx[i].copy()
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        dict_out['timestamp'] = (w_s + w_e) / 2
        if 'imu' in self.modal:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, w_s*self.sr_imu:w_e*self.sr_imu]
            dict_out["imu"] = imu
        if 'audio' in self.modal:
            audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), offset=w_s, duration=w_e-w_s, sr=self.sr_audio)
            dict_out["audio"] = audio
        return dict_out
class Ego4D_Moment(Dataset):
    def __init__(self, folder='../../dataset/ego4d/v2/', window_sec = 1, modality=['imu', 'audio'], split='train'):
        self.folder = folder
        self.modality = modality
        self.metadata = get_ego4d_metadata('../../dataset/ego4d/ego4d.json', "video")
        self.meta_imu = json.load(open('../../dataset/ego4d/v2/annotations/meta_imu.json', 'r'))
        moment_dict = index_moments("../../dataset/ego4d/v2/annotations/moments_{}.json".format(split))

        moment_dict = filter_by_modality(self.metadata, moment_dict, modality)
        self.window_idx = prepare_moment(moment_dict, self.metadata, self.meta_imu, window_sec)
        print(f"Total {len(self.window_idx)} windows to process")

        self.num_class = None
        if 'raw_label' in modality:
            self.labels, self.label_count = self.label_dict()
            self.num_class = len(self.labels)
        if 'close_label' in modality:
            self.labels, self.label_count= self.label_dict()
            self.labels, self.num_class = close_to(self.labels, 'close_{}.txt'.format(split))
        if 'cluster_label' in modality:
            self.labels, self.label_count = self.label_dict()
            # self.labels, self.num_class = cluster_map(self.labels, 'cluster_{}.txt'.for))
    def align_labels(self, labels1, labels2):
        # only for cluster label
        labels = {}
        idx = 0
        for key in labels1:
            if key not in labels:
                labels[key] = idx
                idx += 1
        for key in labels2:
            if key not in labels:
                labels[key] = idx
                idx += 1
        self.labels, self.num_class = cluster_map(labels, 'cluster.txt')
    def __len__(self):
        return len(self.window_idx)
    def label_dict(self):
        label_dict = {}
        label_count = {}
        durations = 0
        for data in self.window_idx:
            label = data['text']
            duration = data['window_end'] - data['window_start']
            durations += duration
            if label not in label_dict:
                label_dict[label] = len(label_dict)
                label_count[label] = 1
            else:
                label_count[label] += 1
        weights = np.array(list(label_count.values()))
        weights = 1 / weights
        weights = weights / weights.sum()
        self.weights = weights.astype(np.float32)
        self.num_class = len(label_dict)
        return label_dict, label_count
    def __getitem__(self, i):
        dict_out = self.window_idx[i]
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        dict_out['label'] = self.labels[dict_out['text']]
        if 'imu' in self.modality:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, w_s*200:w_e*200]
            dict_out["imu"] = imu
        if 'audio' in self.modality:
            audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), offset=w_s, duration=w_e-w_s, sr=16000)
            dict_out["audio"] = audio
        return dict_out
class IMU2CLIP_Dataset(Dataset):
    def __init__(self,  folder='../../dataset/ego4d/v2/', window_sec=2.5, modality=['imu', 'audio'], split='train'):
        self.folder = folder
        self.metadata = get_ego4d_metadata('../../dataset/ego4d/ego4d.json', "video")
        self.meta_imu = json.load(open('../../dataset/ego4d/v2/annotations/meta_imu.json', 'r'))
        self.moments = self.load_csv('dataset_motion_narr_2.5_{}_0.csv'.format(split))
        self.moment_dict = {}
        self.modality = modality
        for i in self.moments:
            if i['video_uid'] not in self.moment_dict:
                self.moment_dict[i['video_uid']] = []
            self.moment_dict[i['video_uid']].append([i['label'], int(i['window_start']), int(i['window_end'])])
        self.moment_dict = filter_by_modality(self.metadata, self.moment_dict, modality)
        # print(f"Total {len(self.moment_dict)} windows to process")
        self.window_idx = prepare_moment(self.moment_dict, self.metadata, self.meta_imu, window_sec)
        print(f"Total {len(self.window_idx)} windows to process")
        self.labels =  {"head movement":0, "stands up":1, "sits down":2, "walking":3}
        self.num_class = len(self.labels)
        self.weights = self.get_weight()
    def load_csv(self, csv_path):
        import csv
        """
        Load a CSV file
        """
        with open(csv_path, "r", encoding="utf-8") as f_name:
            reader = csv.DictReader(f_name)
            data = []
            for row in reader:
                data.append(row)
        return data
    def __len__(self):
        return len(self.window_idx)
    def get_weight(self):
        weight = np.zeros(self.num_class)
        for data in self.window_idx:
            label = self.labels[data['text']]
            weight[label] += 1
        weight = 1 / weight
        weight = weight / weight.sum()
        return weight.astype(np.float32)
    def __getitem__(self, i):
        dict_out = self.window_idx[i]
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        dict_out['label'] = self.labels[dict_out['text']]

        if 'imu' in self.modality:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, w_s*200:w_e*200]
            dict_out["imu"] = imu
        if 'audio' in self.modality:
            audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), offset=w_s, duration=w_e-w_s, sr=16000)
            dict_out["audio"] = audio
        return dict_out
def visualize_data(dict_out):
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wavfile
    fig, axs = plt.subplots(1, 4)
    if 'imu' in dict_out:
        print(dict_out['imu'].shape)
        axs[0].plot(dict_out['imu'][:3].T)
        axs[1].plot(dict_out['imu'][3:].T)
        axs[0].set_title('IMU 1-3')
        axs[1].set_title('IMU 4-6')
    if 'audio' in dict_out:
        axs[2].plot(dict_out['audio'])
        axs[2].set_title('Audio')
        wavfile.write('test.wav', 16000, dict_out['audio'])
    if 'image' in dict_out:
        axs[3].imshow(dict_out['image'])
        axs[3].set_title('Image')
    plt.title(dict_out['text'])
    plt.savefig('test.png')
def visualize_class(labels):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 6})

    values = labels.values()
    labels = labels.keys()
    plt.figure(figsize=(18, 6))
    plt.subplots_adjust(bottom=0.4,)

    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.title('Ego4D Moment Class Distribution')
    plt.xlim(-1, len(values))
    plt.savefig('ego4d_moment_distribution.pdf')
if __name__ == '__main__':
    dataset = Ego4D_Narration(window_sec=1, modality=['audio'])

    data = dataset[0]
    print(data['audio'].shape, data['text'], len(dataset))

    dataset = Ego4D_Narration(window_sec=1, modality=['imu'])

    data = dataset[0]
    print(data['imu'].shape, data['text'], len(dataset))
    # dataset = Ego4D_Moment(window_sec=2.5, modality=['imu', 'raw_label'], split='train')
    # print(dataset.num_class)
    # visualize_class(dataset.labels)

    # for data in dataset:
    # # visualize_data(dataset[1200])
    #     print(data['label'])
    # label_dict = dataset.label_dict()
    # print(label_dict)
    # dataset = IMU2CLIP_Dataset()
    # for data in dataset:
    #     print(data['label'], data['imu'].shape)

