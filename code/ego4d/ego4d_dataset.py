import numpy as np
from torch.utils.data import Dataset
import os
from .extract_imu import get_ego4d_metadata
from .ego4d_utils import index_narrations, index_moments
from tqdm import tqdm
import math
import string
import torch
import json
import pandas as pd
import soundfile as sf
from imagebind import data


def text_process(text):
    text = text.replace("#C C ", "").replace("#C", "").replace("#O", "").replace("#unsure", "").strip().strip(string.punctuation).lower()[:128]
    text = text.replace("summary", "")
    return text

def prepare_windows(narration_dict, metadata, meta_imu, modality, window_sec):
    print("Preparing training data")
    window_idx = []; scenario_map = {}
    for video_uid, narrations in tqdm(narration_dict.items()):
        duration = metadata[video_uid]["video_metadata"]["video_duration_sec"]
        if 'imu' in modality and metadata[video_uid]["has_imu"]:
            duration = min(duration, meta_imu[video_uid])
        for scenario in metadata[video_uid]["scenarios"]:
            if scenario not in scenario_map:
                scenario_map[scenario] = len(scenario_map)
        _scenario = [scenario_map[scenario] for scenario in metadata[video_uid]["scenarios"]]
        for (text, timestamp, frame) in narrations:
            if window_sec is None: # use all the data
                w_s = timestamp - frame/2
                w_e = timestamp + frame/2
            else:
                if timestamp > duration or window_sec > duration:
                    continue # skip if the timestamp is larger than the duration
                if timestamp <= window_sec/2:
                    w_s = 0.0
                    w_e = window_sec
                elif timestamp + window_sec/2 >= duration:
                    w_s = duration - window_sec
                    w_e = duration
                else:
                    w_s = timestamp - window_sec /2
                    w_e = timestamp + window_sec /2
                w_s = int(math.floor(w_s))
                w_e = int(math.floor(w_e))
                try:
                    assert w_e - w_s == window_sec
                except AssertionError:
                    continue
            input_dict = {
                "window_start": w_s,
                "window_end": w_e,
                "video_uid": video_uid,
                "text": text_process(text),
                "scenario": _scenario,
            }
            window_idx.append(input_dict)
    print('Number of Scenario', len(scenario_map))
    scenario_map = {v:k for k,v in scenario_map.items()}
    return window_idx, scenario_map

def narration_summary(narration_dict, summary_dict):
    '''
    narration: {video_uid: [(text, timestamp, frame), ...]}
    summary: {video_uid: [(text, timestamp, frame), ...]}
    for each summary, find the narration that is within the summary time window
    '''
    print("find the match narrations for summary")
    summary_idx = []; avg_narration_len = []
    for video_uid, summaries in tqdm(summary_dict.items()):
        if video_uid not in narration_dict:
            narrations = []
        else:
            narrations = narration_dict[video_uid]
        for (summary, timestamp, frame) in summaries:
            summary_narrations = []
            for (n_text, n_timestamp, n_frame) in narrations:
                if abs(n_timestamp - timestamp) <= frame / 2:
                    summary_narrations.append((n_text, n_timestamp, n_frame))
            summary_idx.append(summary_narrations)
            avg_narration_len.append(len(summary_narrations))
    print('Average narration length in summary:', np.mean(avg_narration_len))
    return summary_idx

def sound_annotation(meta_csv = 'ego4d/action2sound/test_clips_11k.csv', filter_video_uid=[]):
    print("Loading sound annotation from", meta_csv)
    # compatible to \t and ','
    ego4d_sound_meta = pd.read_csv(meta_csv, header=0, sep='\t')
    if 'video_uid' not in ego4d_sound_meta.columns:
        ego4d_sound_meta = pd.read_csv(meta_csv, header=0, sep=',')
    # convert to dict {'video_uid': [start, end, text]}
    narration_dict = {}
    for idx, row in ego4d_sound_meta.iterrows():
        video_uid = row['video_uid']
        if video_uid not in filter_video_uid:
            continue
        narration_time = row['narration_time']
        # clip_text = text_process(row['clip_text'])
        clip_text = row['activity_name']
        positive = row['positive']  # 1 for sound narration, 0 for no sound narration
        if video_uid not in narration_dict:
            narration_dict[video_uid] = []
        narration_dict[video_uid].append((clip_text, narration_time, positive))  # 0 for sound narration
    return narration_dict

def prepare_imu2clip(filter_video_uid):
    annotation_dir = 'ego4d/imu2clip'
    annotation_file = 'dataset_motion_narr_2.5.csv'
    pd_data = pd.read_csv(os.path.join(annotation_dir, annotation_file))
    imu2clip_label = []
    scenario_map = {'moves head': 0, "walk": 1, "sits down": 2, "stands up": 3, "looks up": 0, "looks down": 0, "looks around": 0, 
                    "jumping": 1, "cycling": 1}
    scenario_count = [0] * 4
    for idx, row in pd_data.iterrows():
        video_uid = row['video_uid']
        if video_uid not in filter_video_uid:
            continue
        else:
            scenario_idx = scenario_map[row['label']]
            imu2clip_label.append({"video_uid": video_uid, 
                                   "window_start": row['window_start'], "window_end": row['window_end'],
                                     "scenario":scenario_idx, "text": ""})
            scenario_count[scenario_idx] += 1
    scenario_map = {v:k for k,v in scenario_map.items()}
    return imu2clip_label, scenario_map

def initialization(folder='../dataset/ego4d/v2/', modal=['imu', 'audio']):
    '''
    Initialization function to load metadata and filter video UIDs based on the specified modalities.
    '''
    metadata = get_ego4d_metadata(os.path.join(folder, "ego4d.json"), "video")
    meta_imu = json.load(open(os.path.join(folder, "annotations/meta_imu.json"), 'r'))
    meta_audio = [v[:-4] for v in os.listdir(os.path.join(folder, "audio"))]
    filter_video_uid = []
    for video_uid in list(metadata.keys()):
        if "imu" in modal and video_uid not in meta_imu:
            continue
        if "audio" in modal and video_uid not in meta_audio:
            continue
        filter_video_uid.append(video_uid)
    return metadata, meta_imu, meta_audio, filter_video_uid

def imagebind_wrapper(batch, device='cpu'):
    if 'imu' in batch:
        batch['imu'] = torch.from_numpy(batch['imu'][None, :]).to(device).float()
    if 'audio' in batch:
        batch['audio'] = torch.from_numpy(batch['audio'][None, :])
        batch['audio'] = data.load_and_transform_audio_data([[batch['audio'], 16000]], (device))
    if 'text' in batch:
        batch['text'] = data.load_and_transform_text([batch['text']], device)
    return batch

class Ego4D_Narration(Dataset):
    def __init__(self, folder='../dataset/ego4d/v2/', window_sec = 2, modal=['imu', 'audio', 'narration']):
        self.folder = folder
        self.modal = modal
        self.window_sec = window_sec
        metadata, meta_imu, meta_audio, filter_video_uid = initialization(folder, modal)
        narration_dict, summary_dict = index_narrations(os.path.join(self.folder, "annotations/narration.json"), filter_video_uid)
        if 'summary' in self.modal:
            self.narration_summary = narration_summary(narration_dict, summary_dict)
            self.window_idx, self.scenario_map = prepare_windows(summary_dict, metadata, meta_imu, self.modal, self.window_sec)
            print(len(self.narration_summary), len(self.window_idx))
        else:
            self.window_idx, self.scenario_map = prepare_windows(narration_dict, metadata, meta_imu, self.modal, self.window_sec)
        with open('resources/scenario_map.json', 'w') as f:
            json.dump(self.scenario_map, f, indent=4)
        print(f"There are {len(self.window_idx)} windows to process.")
        self.sr_imu = 200
        self.sr_audio = 16000
    def __len__(self):
        return len(self.window_idx)
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
    
    def __getitem__(self, i, imagebind=False):
        data_sample = self.window_idx[i]
        dict_out = data_sample.copy()
        uid = data_sample["video_uid"]
        w_s = data_sample["window_start"]; w_e = data_sample["window_end"]
        scenario_vec = np.zeros(len(self.scenario_map), dtype=float)
        scenario = ', '.join([self.scenario_map[s].replace('/', ' or ') for s in dict_out['scenario']])
        dict_out['scenario_name'] = scenario
        scenario_vec[dict_out['scenario']] = 1
        dict_out['scenario'] = scenario_vec
        if 'imu' in self.modal:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, int(w_s*self.sr_imu): int(w_e*self.sr_imu)]
            if self.window_sec is not None and imu.shape[-1] < self.sr_imu * self.window_sec:
                imu = np.pad(imu, ((0,0), (0, self.sr_imu * self.window_sec - imu.shape[-1])))
            dict_out["imu"] = imu
        if 'audio' in self.modal:
            # audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), 
            #                         offset=w_s, duration=self.window_sec, sr=self.sr_audio)
            audio, sr = sf.read(os.path.join(self.folder, 'audio', f"{uid}.mp3"), start=int(w_s*self.sr_audio), stop=int(w_e*self.sr_audio), 
                                always_2d=True, fill_value=0, dtype='float32')
            audio = audio[:, 0]
            if self.window_sec is not None and len(audio) < self.sr_audio * self.window_sec:
                audio = np.pad(audio, (0, self.sr_audio * self.window_sec - audio.shape[-1]))
            dict_out["audio"] = audio
        if imagebind:
            dict_out = imagebind_wrapper(dict_out)
        return dict_out

class Ego4D_Action2Sound(Ego4D_Narration):
    def __init__(self, folder='../dataset/ego4d/v2/', window_sec=2, modal=['imu', 'audio'], csv='ego4d/action2sound/test_clips_11k.csv'):
        self.folder = folder
        self.window_sec = window_sec
        self.modal = modal

        metadata, meta_imu, meta_audio, filter_video_uid = initialization(folder, modal)
        if 'narration' in self.modal:
            narration_dict = sound_annotation(csv, filter_video_uid)
            self.window_idx, self.scenario_map = prepare_windows(narration_dict, metadata, meta_imu, modal, window_sec)
        elif 'summary' in self.modal:
            narration_dict = sound_annotation(csv, filter_video_uid)
            _, summary_dict = index_narrations(os.path.join(self.folder, "annotations/narration.json"), filter_video_uid)
            self.narration_summary = narration_summary(narration_dict, summary_dict)
            self.window_idx, self.scenario_map = prepare_windows(summary_dict, metadata, meta_imu, self.modal, self.window_sec)
        else:
            return NotImplementedError("Modalities not supported: {}".format(self.modal))

        print(f"There are {len(self.window_idx)} windows to process.")
        self.sr_imu = 200
        self.sr_audio = 16000

class Ego4D_Moment(Ego4D_Narration):
    def __init__(self, folder='../dataset/ego4d/v2/', window_sec=2, modal=['imu', 'audio'], split='train'):
        self.folder = folder
        self.window_sec = window_sec
        self.modal = modal

        metadata = get_ego4d_metadata(os.path.join(self.folder, "ego4d.json"), "video")
        meta_imu = json.load(open(os.path.join(self.folder, "annotations/meta_imu.json"), 'r'))
        meta_audio = [v[:-4] for v in os.listdir(os.path.join(self.folder, "audio"))]
        filter_video_uid = []
        for video_uid in list(metadata.keys()):
            if "imu" in modal and video_uid not in meta_imu:
                continue
            if "audio" in modal and video_uid not in meta_audio:
                continue
            filter_video_uid.append(video_uid)     
        moment_dict = index_moments(os.path.join(self.folder, "annotations/moments_{}.json".format(split)), filter_video_uid)    
        self.window_idx, self.scenario_map = prepare_windows(moment_dict, metadata, meta_imu, modal, window_sec)
        print(f"There are {len(self.window_idx)} windows to process.")
        self.sr_imu = 200
        self.sr_audio = 16000
    def __len__(self):
        return len(self.window_idx)
    
class Ego4D_Narration_Sequence(Ego4D_Narration):
    '''
    Building upon Ego4D_Narration, this class is used to generate a sequence of data for each scenario
    The class accept pre-definded Ego4D_Narration to initialize it
    '''
    def __init__(self, parent_obj, num_sequence=5):
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
        dict_out = {'audio':[], 'imu':[]}    
        for idx in sequence['window_idx']:
            _dict = super().__getitem__(idx)
            for key in dict_out.keys():
                dict_out[key].append(_dict[key])
        dict_out = {k: np.stack(dict_out[k]) for k in dict_out.keys()}
        scenario_vec = np.zeros(len(self.scenario_map), dtype=np.float32)
        scenario_vec[sequence['scenario']] = 1
        dict_out['scenario'] = scenario_vec
        return dict_out

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

class Ego4D_IMU2CLIP(Ego4D_Narration):
    def __init__(self, folder='../dataset/ego4d/v2/', window_sec=5, modal=['imu', 'audio']):
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
        self.window_idx, self.scenario_map = prepare_imu2clip(filter_video_uid)
        self.sr_imu = 200
        self.sr_audio = 16000
    def __len__(self):
        return len(self.window_idx)
