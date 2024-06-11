import numpy as np
from torch.utils.data import Dataset
import os
from extract_imu import get_ego4d_metadata
from utils import index_narrations, index_moments
from tqdm import tqdm
import math
import string
import librosa
import json
from tqdm import tqdm
from code.utils.text_cluster import close_to, cluster_plot, cluster_map

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
def filter_by_modality(metadata, data_dict, modality: list):
    new_data_dict = {}
    # binaural_audio_meta = os.listdir('../dataset/ego4d/v2/components/binaural_audio')
    # print(f"Total {len(binaural_audio_meta)} binaural audio files.")
    # for b in binaural_audio_meta:
    #     if b in metadata:
    #         print(b)
    #     if b in data_dict:
    #         print(b)
    for video_uid, data in data_dict.items():
        if video_uid not in metadata:
            continue
        if "imu" in modality and not metadata[video_uid]["has_imu"]:        
            continue
        if "audio" in modality and metadata[video_uid]["video_metadata"]["audio_duration_sec"] is None:
            continue
        # if "binaural_audio" in modality and (video_uid + '.wav') not in binaural_audio_meta:
        #     continue
        new_data_dict[video_uid] = data
    return new_data_dict
def prepare_narration(narration_dict, metadata, meta_imu, window_sec):
    window_idx = []
    for video_uid, narrations in (narration_dict.items()):
        if not metadata[video_uid]["has_imu"] or not video_uid in meta_imu:
            continue
        video_duration = metadata[video_uid]["video_metadata"]["video_duration_sec"]
        imu_duration = meta_imu[video_uid]
        duration = min(video_duration, imu_duration) 
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
                "narration_uid": a_uid,
                "text": clean_narration_text(text),
            }
            window_idx.append(input_dict)
    print(f"There are {len(window_idx)} windows to process.")
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
    def __init__(self, folder='', window_sec = 1, modality=['imu', 'audio'], split='train'):
        self.folder = folder
        self.modality = modality
        self.metadata = get_ego4d_metadata('../dataset/ego4d/v2/annotations/ego4d.json', "video")
        self.meta_imu = json.load(open('../dataset/ego4d/v2/annotations/meta_imu.json', 'r'))
        narration_dict, _ = index_narrations()
        narration_dict = filter_by_modality(self.metadata, narration_dict, modality)
        self.window_idx = prepare_narration(narration_dict, self.metadata, self.meta_imu, window_sec)
        if split == 'train':
            self.window_idx = self.window_idx[:int(len(self.window_idx) * 0.8)]
        elif split == 'val':
            self.window_idx = self.window_idx[int(len(self.window_idx) * 0.8): int(len(self.window_idx) * 0.9)]
        else:
            self.window_idx = self.window_idx[int(len(self.window_idx) * 0.9):]
    def __len__(self):
        return len(self.window_idx)       
    def __getitem__(self, i):
        dict_out = self.window_idx[i]
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        text = dict_out["text"]
        if 'imu' in self.modality:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, w_s*200:w_e*200]
            dict_out["imu"] = imu
        if 'audio' in self.modality:
            audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), offset=w_s, duration=w_e-w_s, sr=16000)
            dict_out["audio"] = audio
        dict_out["narration"] = text
        return dict_out
class Ego4D_Moment(Dataset):
    def __init__(self, folder='../dataset/ego4d/v2/', window_sec = 1, modality=['imu', 'audio'], split='train'):
        self.folder = folder
        self.modality = modality
        self.metadata = get_ego4d_metadata('../dataset/ego4d/ego4d.json', "video")
        self.meta_imu = json.load(open('../dataset/ego4d/v2/annotations/meta_imu.json', 'r'))
        moment_dict_train = index_moments("../dataset/ego4d/v2/annotations/moments_train.json".format(split))
        moment_dict_val = index_moments("../dataset/ego4d/v2/annotations/moments_train.json".format(split))
        moment_dict = {**moment_dict_train, **moment_dict_val}

        moment_dict = filter_by_modality(self.metadata, moment_dict, modality)
        self.window_idx = prepare_moment(moment_dict, self.metadata, self.meta_imu, window_sec)
        print(f"Total {len(self.window_idx)} windows to process")

        self.num_class = None
        if 'raw_label' in modality:
            self.labels = self.label_dict()
            self.num_class = len(self.labels)
        if 'close_label' in modality:
            # if os.path.exists('close_label.txt'.format(split)):
            #     self.labels = json.load(open('close_label.txt'.format(split), 'r'))
            #     self.num_class = max(self.labels.values()) + 1
            # else: # do it on-the-fly
            self.labels = self.label_dict()
            self.labels, self.num_class = close_to(self.labels, 'close_label.txt'.format(split))
        if 'cluster_label' in modality:
            # if os.path.exists('cluster_label.txt'.format(split)):
            #     self.labels = json.load(open('cluster_label.txt'.format(split), 'r'))
            #     self.num_class = max(self.labels.values()) + 1
            # else:
            self.labels = self.label_dict()
            self.labels, self.num_class = cluster_map(self.labels, 'cluster_label.txt'.format(split))
        if split == 'train':
            self.window_idx = self.window_idx[:int(len(self.window_idx) * 0.8)]
        elif split == 'val':
            self.window_idx = self.window_idx[int(len(self.window_idx) * 0.8):]
    def __len__(self):
        return len(self.window_idx)
    def label_dict(self):
        label_dict = {}
        durations = 0
        for data in self.window_idx:
            label = data['text']
            duration = data['window_end'] - data['window_start']
            durations += duration
            if label not in label_dict:
                label_dict[label] = len(label_dict) + 1
            else:
                label_dict[label] += 1
        # sort it
        label_dict = dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=True))
        # label_dict = {k: i for i, (k, v) in enumerate(label_dict.items())}
        print(f"Total {durations} seconds of data, average {durations / len(self.window_idx)} seconds per label.")
        return label_dict       
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
    def __init__(self,  folder='../dataset/ego4d/v2/', window_sec=2.5, modality=['imu', 'audio'], split='train'):
        self.folder = folder
        self.metadata = get_ego4d_metadata('../dataset/ego4d/ego4d.json', "video")
        self.meta_imu = json.load(open('../dataset/ego4d/v2/annotations/meta_imu.json', 'r'))
        self.moments = self.load_csv('dataset_motion_narr_2.5_{}_0.csv'.format(split))
        self.moment_dict = {}
        self.modality = modality
        for i in self.moments:
            if i['video_uid'] not in self.moment_dict:
                self.moment_dict[i['video_uid']] = []
            self.moment_dict[i['video_uid']].append([i['label'], int(i['window_start']), int(i['window_end'])])
        self.moment_dict = filter_by_modality(self.metadata, self.moment_dict, modality)
        print(f"Total {len(self.moment_dict)} windows to process")
        self.window_idx = prepare_moment(self.moment_dict, self.metadata, self.meta_imu, window_sec)
        print(f"Total {len(self.window_idx)} windows to process")
        self.labels =  {"head movement":0, "stands up":1, "sits down":2, "walking":3}
        self.num_class = len(self.labels)
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
    fig, axs = plt.subplots(4)
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
    # dataset = Ego4D_Narration(window_sec=5, modality=['binaural_audio'])
    # data = dataset[0]
    # print(data['imu'].shape, data['audio'].shape, print(data['text']))

    dataset = Ego4D_Moment(window_sec=2.5, modality=['imu', 'raw_label'], split='train')
    print(dataset.num_class)
    visualize_class(dataset.labels)

    # for data in dataset:
    # # visualize_data(dataset[1200])
    #     print(data['label'])
    # label_dict = dataset.label_dict()
    # print(label_dict)
    # dataset = IMU2CLIP_Dataset()
    # for data in dataset:
    #     print(data['label'], data['imu'].shape)
