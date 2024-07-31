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
    scenario_map = {v:k for k,v in scenario_map.items()}
    return window_idx, scenario_map
def prepare_free(filter_video_uid, metadata, meta_audio, meta_imu, modality, window_sec):
    window_idx = []
    scenario_map = {}
    for video_uid in tqdm(filter_video_uid):
        duration = metadata[video_uid]["video_metadata"]["video_duration_sec"]
        if 'imu' in modality and metadata[video_uid]["has_imu"]:
            duration = min(duration, meta_imu[video_uid])
        for scenario in metadata[video_uid]["scenarios"]:
            if scenario not in scenario_map:
                scenario_map[scenario] = len(scenario_map)
        _scenario = [scenario_map[scenario] for scenario in metadata[video_uid]["scenarios"]]
        num_sample = int(duration // window_sec)
        for i in range(num_sample):
            w_s = i * window_sec
            w_e = (i+1) * window_sec
            input_dict = {
                "window_start": w_s,
                "window_end": w_e,
                "video_uid": video_uid,
                "text": '',
                "scenario": _scenario,
            }
            window_idx.append(input_dict)
    print('Number of Scenario', len(scenario_map))
    scenario_map = {v:k for k,v in scenario_map.items()}
    return window_idx, scenario_map

class Ego4D_Narration(Dataset):
    def __init__(self, pre_compute_json=None, folder='../dataset/ego4d/v2/', window_sec = 2, modal=['imu', 'audio']):
        self.folder = folder
        self.modal = modal
        self.window_sec = window_sec
        if pre_compute_json is not None:
            self.pre_compute_json = pre_compute_json
            with open(pre_compute_json, 'r') as f:
                self.window_idx = json.load(f)
            self.scenario_map = json.load(open('resources/scenario_map.json', 'r'))
            # change the key of sceanrio_map from string to int
            self.scenario_map = {int(k):v for k,v in self.scenario_map.items()}

        else:
            metadata = get_ego4d_metadata(os.path.join(self.folder, "ego4d.json"), "video")
            meta_imu = json.load(open(os.path.join(self.folder, "annotations/meta_imu.json"), 'r'))
            meta_audio = [v[:-4] for v in os.listdir(os.path.join(self.folder, "audio"))]
            filter_video_uid = []
            for video_uid in list(metadata.keys()):
                keep_or_not = True
                if "imu" in modal and video_uid not in meta_imu:
                    keep_or_not = False
                if "audio" in modal and video_uid not in meta_audio:
                    keep_or_not = False
                if keep_or_not:
                    filter_video_uid.append(video_uid)
            narration_dict = index_narrations(os.path.join(self.folder, "annotations/narration.json"), filter_video_uid)
            self.window_idx, self.scenario_map = prepare_narration(narration_dict, metadata, meta_audio, meta_imu, self.modal, window_sec)
            # save scenario_map
            with open('resources/scenario_map.json', 'w') as f:
                json.dump(self.scenario_map, f, indent=4)
        print(f"There are {len(self.window_idx)} windows to process.")
        self.sr_imu = 200
        self.sr_audio = 16000
    def audio_stat(self, fname='resources/egoexo_atomic_prune.json'):
        for i in tqdm(range(0, self.__len__())):
            data = self.__getitem__(i)
            audio = data['audio']
            snr = np.max(audio) / np.mean(np.abs(audio))
            self.window_idx[i]['snr'] = float(snr)
            # rms = np.sqrt(np.mean(audio**2))
            # self.window_idx[i]['rms'] = float(rms)
        self.save_json(fname)
    def audio_prune(self, fname='resources/egoexo_atomic_prune.json', snr_thres=None,):
        new_idx = []
        for i in tqdm(range(0, self.__len__())):
            if self.window_idx[i]['snr'] > snr_thres:
                new_idx.append(i)
            # data = self.__getitem__(i)
            # if data['snr'] > snr_thres:
            #     new_idx.append(i)
        self.window_idx = [self.window_idx[i] for i in new_idx]
        print(f"remaining {len(self.window_idx)} windows")
        self.save_json(fname)
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
        if len(value) > 1: # add multiple values
            for i, v in zip(idx, value):
                self.window_idx[i][key] = float(v)
        else:
            self.window_idx[idx][key] = float(value)
    def split_with_scenario(self, ratio = 0.8):
        '''
        For each video (may refer to multiple scenarios), split the data into train and test set
        For the fairness of classification
        '''
        train_idx = []
        test_idx = []
        video_uid_idx = {}
        for i, data in enumerate(self.window_idx):
            video_uid = data['video_uid']
            if video_uid not in video_uid_idx:
                video_uid_idx[video_uid] = []
            video_uid_idx[video_uid].append(i)
        
        for video_uid, idx in video_uid_idx.items():
            train_size = int(len(idx) * ratio)
            train_idx += idx[:train_size]
            test_idx += idx[train_size:]
        print('Train size: {}, Test size: {}'.format(len(train_idx), len(test_idx)))
        return train_idx, test_idx
    def split_new_scenario(self, novel_scenario):
        '''
        In case we want to discover new scenario with the dataset, the setting is closed to continual learning
        '''
        scenario_idx = {}
        for i, data in enumerate(self.window_idx):
            scenarios = data['scenario']
            for scenario in scenarios:
                if scenario not in scenario_idx:
                    scenario_idx[scenario] = []
                scenario_idx[scenario].append(i)
        # select the last scenarios
        novel_scenario_names = [self.scenario_map[scenario] for scenario in novel_scenario]
        print('Selected scenario: {}'.format(novel_scenario_names))
        novel_idx = []
        for scenario in novel_scenario:
            novel_idx += scenario_idx[scenario]
        # remove duplicate
        novel_idx = list(set(novel_idx))
        support_idx = list(set(range(len(self.window_idx))) - set(novel_idx))
        print('Support size: {}, Novel size: {}'.format(len(support_idx), len(novel_idx)))
        return support_idx, novel_idx

    def __getitem__(self, i):
        dict_out = self.window_idx[i].copy()
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        dict_out['timestamp'] = (w_s + w_e) / 2

        scenario_vec = np.zeros(len(self.scenario_map), dtype=float)
        scenario_vec[dict_out['scenario']] = 1
        dict_out['scenario'] = scenario_vec
        if 'imu' in self.modal:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, int(w_s*self.sr_imu): int(w_e*self.sr_imu)]
            dict_out["imu"] = imu
        if 'audio' in self.modal:
            if 'context_audio' in self.modal: # no need to load twice
                add_width = 2
                audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), 
                                     offset=w_s-add_width, duration=self.window_sec+2*add_width, sr=self.sr_audio)
                if audio.shape[-1] < self.sr_audio * (self.window_sec + add_width*2):
                    audio = np.pad(audio, (0, self.sr_audio * (self.window_sec + add_width*2) - audio.shape[-1]))
                dict_out["context_audio"] = audio
                audio = audio[add_width*self.sr_audio:-add_width*self.sr_audio]
                dict_out["audio"] = audio
            else:
                audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), 
                                        offset=w_s, duration=self.window_sec, sr=self.sr_audio)
                if audio.shape[-1] < self.sr_audio * self.window_sec:
                    audio = np.pad(audio, (0, self.sr_audio * self.window_sec - audio.shape[-1]))
                dict_out["audio"] = audio
        return dict_out

class Ego4D_Narration_Sequence(Ego4D_Narration):
    '''
    Building upon Ego4D_Narration, this class is used to generate a sequence of data for each scenario
    The class accept pre-definded Ego4D_Narration to initialize it
    '''
    def __init__(self, parent_obj, num_sequence=5):
        if isinstance(parent_obj, Ego4D_Narration):
            pass
        else:
            parent_obj = parent_obj.dataset
        for key in parent_obj.__dict__.keys():
            setattr(self, key, getattr(parent_obj, key))
        sequences = []
        for i in range(0, super().__len__() - num_sequence, num_sequence//2):
            video_uid = self.window_idx[i]['video_uid']
            sequence = [i]
            for j in range(1, num_sequence):
                _video_uid = self.window_idx[i + j]['video_uid']
                if video_uid != _video_uid:
                    break
                sequence.append(i + j)
            if len(sequence) == num_sequence:
                sequences.append({'window_idx':sequence, 'scenario':self.window_idx[i]['scenario'], 'video_uid':video_uid})
        self.sequences = sequences       
        print('Total {} sequences'.format(len(self.sequences))) 
    def split_with_scenario(self, ratio = 0.8):
        train_idx = []
        test_idx = []
        video_uid_idx = {}
        for i, data in enumerate(self.sequences):
            video_uid = data['video_uid']
            if video_uid not in video_uid_idx:
                video_uid_idx[video_uid] = []
            video_uid_idx[video_uid].append(i)
        
        for video_uid, idx in video_uid_idx.items():
            train_size = int(len(idx) * ratio)
            train_idx += idx[:train_size]
            test_idx += idx[train_size:]
        print('Train size: {}, Test size: {}'.format(len(train_idx), len(test_idx)))
        return train_idx, test_idx
    def split_new_scenario(self, novel_scenario):
        '''
        In case we want to discover new scenario with the dataset, the setting is closed to continual learning
        '''
        scenario_idx = {}
        for i, data in enumerate(self.sequences):
            scenarios = data['scenario']
            for scenario in scenarios:
                if scenario not in scenario_idx:
                    scenario_idx[scenario] = []
                scenario_idx[scenario].append(i)
        # select the last scenarios
        novel_scenario_names = [self.scenario_map[scenario] for scenario in novel_scenario]
        print('Selected scenario: {}'.format(novel_scenario_names))
        novel_idx = []
        for scenario in novel_scenario:
            novel_idx += scenario_idx[scenario]
        # remove duplicate
        novel_idx = list(set(novel_idx))
        support_idx = list(set(range(len(self.sequences))) - set(novel_idx))
        print('Support size: {}, Novel size: {}'.format(len(support_idx), len(novel_idx)))
        return support_idx, novel_idx

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, i):
        sequence = self.sequences[i]
        dict_out = {'audio':[], 'imu':[],}    
        for idx in sequence['window_idx']:
            _dict = super().__getitem__(idx)
            for key in dict_out.keys():
                dict_out[key].append(_dict[key])
        dict_out = {k: np.stack(dict_out[k]) for k in dict_out.keys()}
        scenario_vec = np.zeros(len(self.scenario_map), dtype=np.float32)
        scenario_vec[sequence['scenario']] = 1
        dict_out['scenario'] = scenario_vec
        return dict_out

class Ego4D_Free(Ego4D_Narration):
    def __init__(self, folder='../dataset/ego4d/v2/', window_sec=20, modal=['imu', 'audio']):
        self.folder = folder
        self.window_sec = window_sec
        self.modal = modal
        metadata = get_ego4d_metadata(os.path.join(self.folder, "ego4d.json"), "video")
        meta_imu = json.load(open(os.path.join(self.folder, "annotations/meta_imu.json"), 'r'))
        meta_audio = [v[:-4] for v in os.listdir(os.path.join(self.folder, "audio"))]
        filter_video_uid = []
        for video_uid in list(metadata.keys()):
            keep_or_not = True
            if "imu" in modal and video_uid not in meta_imu:
                keep_or_not = False
            if "audio" in modal and video_uid not in meta_audio:
                keep_or_not = False
            if keep_or_not:
                filter_video_uid.append(video_uid)
        self.window_idx, self.scenario_map = prepare_free(filter_video_uid, metadata, meta_audio, meta_imu, modal, window_sec)
        print(f"There are {len(self.window_idx)} windows to process.")
        self.sr_imu = 200
        self.sr_audio = 16000
    def __len__(self):
        return len(self.window_idx)


if __name__ == '__main__':
    dataset = Ego4D_Narration(window_sec=1, folder='../../dataset/ego4d/v2/', modal=['audio', 'imu'])
    for i in range(10):
        data = dataset[i]
        print(data['audio'].shape, data['imu'].shape, data['text'], len(dataset))

