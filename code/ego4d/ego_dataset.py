import numpy as np
from torch.utils.data import Dataset
import os
from .extract_imu import get_ego4d_metadata
from .ego4d_utils import index_narrations, index_moments
from tqdm import tqdm
import math
import string
import librosa
import soundfile as sf
import json
from tqdm import tqdm

from .text_cluster import close_to, cluster_plot, cluster_map
import time

def prepare_narration(narration_dict, metadata, meta_audio, meta_imu, modality, window_sec):
    window_idx = []
    scenario_map = {}
    for video_uid, narrations in tqdm(narration_dict.items()):
        duration = metadata[video_uid]["video_metadata"]["video_duration_sec"]
        if 'imu' in modality and metadata[video_uid]["has_imu"]:
            duration = min(duration, meta_imu[video_uid])
        for scenario in metadata[video_uid]["scenarios"]:
            if scenario not in scenario_map:
                scenario_map[scenario] = len(scenario_map)
        _scenario = [scenario_map[scenario] for scenario in metadata[video_uid]["scenarios"]]
        for (timestamp, text, a_uid, _) in narrations:
            if timestamp > duration or window_sec > duration:
                continue # skip if the timestamp is larger than the duration
            elif '#c' not in text.lower():
                continue
            else:
                # check if it's the timestamp is at the very beginning
                if timestamp <= window_sec:
                    w_s = 0.0
                    w_e = window_sec
                # check if it's the time stamp is at the very end
                elif timestamp + window_sec >= duration:
                    w_s = duration - window_sec
                    w_e = duration
                # else get a window of data around the timestamps
                else:
                    w_s = timestamp - window_sec /2 
                    w_e = timestamp + window_sec /2
            w_s = int(math.floor(w_s))
            w_e = int(math.floor(w_e))
            try:
                assert w_e - w_s == window_sec
            except AssertionError:
                continue
            # sub_map = {"#o":0, "#c":1, "#unsure":2}
            # subject = 2
            # for i, sub in enumerate(sub_map.keys()):
            #     if sub in text.lower():
            #         subject = sub_map[sub]
            input_dict = {
                # "timestamp": (w_s + w_e)/2,
                "window_start": w_s,
                "window_end": w_e,
                "video_uid": video_uid,
                "text": text
                        .replace("#C C ", "")
                        .replace("#C", "")
                        .replace("#O", "")
                        .replace("#unsure", "")
                        .strip()
                        .strip(string.punctuation)
                        .lower()[:128]
                        ,
                # "subject": subject,
                "scenario": _scenario,
                # "scenario_name": metadata[video_uid]["scenarios"]
            }
            window_idx.append(input_dict)
    print('Number of Scenario', len(scenario_map))
    return window_idx
class Ego4D_Narration(Dataset):
    def __init__(self, pre_compute_json=None, folder='../dataset/ego4d/v2/', window_sec = 2, modal=['imu', 'audio']):
        self.folder = folder
        self.modal = modal
        self.window_sec = window_sec
        if pre_compute_json is not None:
            self.pre_compute_json = pre_compute_json
            with open(pre_compute_json, 'r') as f:
                self.window_idx = json.load(f)
        else:
            metadata = get_ego4d_metadata(os.path.join(self.folder, "ego4d.json"), "video")
            meta_imu = json.load(open(os.path.join(self.folder, "annotations/meta_imu.json"), 'r'))
            meta_audio = [v[:-4] for v in os.listdir(os.path.join(self.folder, "audio"))]
            filter_video_uid = []
            for video_uid in list(metadata.keys()):
                keep_or_not = True
                if "imu" in modal:
                    if not metadata[video_uid]["has_imu"] and video_uid not in meta_imu:
                        keep_or_not = False
                if "audio" in modal:
                    if video_uid not in meta_audio:
                        keep_or_not = False
                if keep_or_not:
                    filter_video_uid.append(video_uid)
            narration_dict = index_narrations(os.path.join(self.folder, "annotations/narration.json"), filter_video_uid)
            self.window_idx = prepare_narration(narration_dict, metadata, meta_audio, meta_imu, self.modal, window_sec)
        print(f"There are {len(self.window_idx)} windows to process.")
        # self.subject_weight = self.get_class_weight('subject')
        # self.scenario_weight = self.get_class_weight('scenario')
        self.sr_imu = 200
        self.sr_audio = 16000
    def get_class_weight(self, key):
        label_count = {}
        for data in self.window_idx:
            label = data[key]
            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1
        label_count = [k/sum(label_count.values()) for k in label_count.values()]
        self.class_weight = label_count
    def __len__(self):
        return len(self.window_idx)
    def save_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.window_idx, f, indent=4)    
    def add(self, idx, key, value):
        self.window_idx[idx][key] = value

    def split_with_scenario(self, ratio = 0.8):
        train_idx = []
        test_idx = []
        scenario_idx = {}
        for i, data in enumerate(self.window_idx):
            for scenario in data['scenario']:
                if scenario not in scenario_idx:
                    scenario_idx[scenario] = []
                scenario_idx[scenario].append(i)
        
        for scenario, idx in scenario_idx.items():
            train_size = int(len(idx) * ratio)
            train_idx += idx[:train_size]
            test_idx += idx[train_size:]
        return train_idx, test_idx
    
    def __getitem__(self, i):
        dict_out = self.window_idx[i].copy()
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        dict_out['timestamp'] = (w_s + w_e) / 2

        scenario_vec = np.zeros(91, dtype=float)
        scenario_vec[dict_out['scenario']] = 1
        dict_out['scenario'] = scenario_vec
        if 'imu' in self.modal:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, w_s*self.sr_imu:w_e*self.sr_imu]
            dict_out["imu"] = imu
        if 'audio' in self.modal:
            audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), 
                                     offset=w_s, duration=self.window_sec, sr=self.sr_audio)
            if audio.shape[-1] < self.sr_audio * self.window_sec:
                audio = np.pad(audio, (0, self.sr_audio * self.window_sec - audio.shape[-1]))
            dict_out["audio"] = audio
        if 'context_audio' in self.modal:
            context_length = 2
            offset = max(np.random.randint(-context_length, context_length) + w_s, 0)
            audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), offset=offset, duration=context_length, sr=self.sr_audio)
            if audio.shape[-1] < self.sr_audio * context_length:
                audio = np.pad(audio, (0, self.sr_audio * context_length - audio.shape[-1]))
            dict_out["context_audio"] = audio
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
    dataset = Ego4D_Narration(window_sec=1, folder='../../dataset/ego4d/v2/', modal=['audio', 'imu'])
    for i in range(10):
        data = dataset[i]
        print(data['audio'].shape, data['imu'].shape, data['text'], len(dataset))

