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
import pandas as pd

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
def prepare_moment(moment_dict, metadata, meta_audio, meta_imu, modality, window_sec):
    '''
    momentdict = ['video_uid':[(moment, start, end), (moment, start, end)]]
    '''
    window_idx = []; scenario_map = {}
    for video_uid, moments in tqdm(moment_dict.items()):
        duration = metadata[video_uid]["video_metadata"]["video_duration_sec"]
        for scenario in metadata[video_uid]["scenarios"]:
            if scenario not in scenario_map:
                scenario_map[scenario] = len(scenario_map)
        _scenario = [scenario_map[scenario] for scenario in metadata[video_uid]["scenarios"]]
        if 'imu' in modality and metadata[video_uid]["has_imu"]:
            duration = min(duration, meta_imu[video_uid])
        for moment, start, end in moments:
            if start > duration or window_sec > duration:
                continue # skip if the timestamp is larger than the duration
            elif start <= window_sec:
                w_s = 0.0
                w_e = window_sec
            elif start + window_sec >= duration:
                w_s = duration - window_sec
                w_e = duration
            else:
                w_s = start - window_sec /2
                w_e = start + window_sec /2
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
                "text": moment,
                "scenario": _scenario
            }
            window_idx.append(input_dict)
    return window_idx, scenario_map

class Ego4D_Narration(Dataset):
    def __init__(self, pre_compute_json=None, folder='../dataset/ego4d/v2/', window_sec = 2, modal=['imu', 'audio', 'capture24']):
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
                if "imu" in modal and video_uid not in meta_imu:
                    continue
                if "audio" in modal and video_uid not in meta_audio:
                    continue
                filter_video_uid.append(video_uid)
            narration_dict = index_narrations(os.path.join(self.folder, "annotations/narration.json"), filter_video_uid)
            self.window_idx, self.scenario_map = prepare_narration(narration_dict, metadata, meta_audio, meta_imu, self.modal, window_sec)
            # save scenario_map
            with open('resources/scenario_map.json', 'w') as f:
                json.dump(self.scenario_map, f, indent=4)
            self.capture24 = np.load('./resources/ego4d_local_mapping.npy')
        print(f"There are {len(self.window_idx)} windows to process.")
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
        data_samele = self.window_idx[i]
        dict_out = {}
        uid = data_samele["video_uid"]
        w_s = data_samele["window_start"]
        w_e = data_samele["window_end"]
        dict_out['timestamp'] = (w_s + w_e) / 2
        dict_out['scenario'] = data_samele['scenario']
        dict_out['text'] = [data_samele['text']]

        scenario_vec = np.zeros(len(self.scenario_map), dtype=float)
        scenario_vec[dict_out['scenario']] = 1
        dict_out['scenario'] = scenario_vec
        if 'imu' in self.modal:
            imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
            imu = imu[:, int(w_s*self.sr_imu): int(w_e*self.sr_imu)]
            if imu.shape[-1] < self.sr_imu * self.window_sec:
                imu = np.pad(imu, ((0,0), (0, self.sr_imu * self.window_sec - imu.shape[-1])))
            dict_out["imu"] = imu
        if 'audio' in self.modal:
            audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), 
                                    offset=w_s, duration=self.window_sec, sr=self.sr_audio)
            if audio.shape[-1] < self.sr_audio * self.window_sec:
                audio = np.pad(audio, (0, self.sr_audio * self.window_sec - audio.shape[-1]))
            dict_out["audio"] = audio
        if 'capture24' in self.modal:
            dict_out['capture24'] = self.capture24[:, i]
        return dict_out

class Ego4D_Sound(Ego4D_Narration):
    def __init__(self, meta_csv='ego4d/train_clips_1.2m.csv', folder='../dataset/ego4d/v2/', 
                 window_sec = 2, modal=['imu', 'audio']):
        '''
        The pre_compute_json is the csv file from Ego4D_Sounds (https://github.com/Ego4DSounds/Ego4DSounds)
        Convert it into Ego4D_Narration format.
        (wind)
        'window_start', 'window_end', 'video_uid', 'scenario', 'text'
        '''
        self.folder = folder
        self.window_sec = window_sec
        self.modal = modal
        self.window_idx = []
        # self.scenario_map = json.load(open('resources/scenario_map.json', 'r'))
        # self.scenario_map = {int(k):v for k,v in self.scenario_map.items()}
        # self.scenario_convert_map = {v:k for k,v in self.scenario_map.items()}

        ego4d_sound_meta = pd.read_csv(meta_csv, header=None, delimiter='\t', on_bad_lines='skip', usecols=[0, 4, 7,], skiprows=1)
        metadata = get_ego4d_metadata(os.path.join(self.folder, "ego4d.json"), "video")
        meta_imu = json.load(open(os.path.join(self.folder, "annotations/meta_imu.json"), 'r'))
        meta_audio = [v[:-4] for v in os.listdir(os.path.join(self.folder, "audio"))]

        self.scenario_map = {}
        for idx, row in ego4d_sound_meta.iterrows():
            row = row.values
            video_uid = row[0]
            if video_uid not in metadata:
                continue
            scenarios = metadata[video_uid]['scenarios']
            for scenario in scenarios:
                if scenario not in self.scenario_map:
                    self.scenario_map[scenario] = len(self.scenario_map)
            scenario = [self.scenario_map[s] for s in scenarios]
            time_stamp = row[1]
            if video_uid not in meta_imu and 'imu' in modal:
                continue 
            if video_uid not in meta_audio and 'audio' in modal:
                continue
            if time_stamp + window_sec/2 > meta_imu[video_uid] or time_stamp <= window_sec/2:
                continue
            self.window_idx.append(
                {
                # 'window_start': int(math.floor(time_stamp - window_sec/2)),
                #  'window_end': int(math.floor(time_stamp + window_sec/2)), 
                 'window_start': time_stamp - window_sec/2,
                 'window_end': time_stamp + window_sec/2, 
                 'video_uid': video_uid, 
                 'text': row[2].replace("#C C ", "")
                        .replace("#C", "")
                        .replace("#O", "")
                        .replace("#unsure", "")
                        .strip()
                        .strip(string.punctuation)
                        .lower()[:128],
                 'scenario': scenario       
                }
            )
            # if idx > 100:
            #     break
        print(f"Total {len(self.window_idx)} windows")
        self.scenario_map = {v:k for k,v in self.scenario_map.items()}
        print('Total scenarios:', len(self.scenario_map))
        self.sr_audio = 16000
        self.sr_imu = 200

class Ego4D_Narration_Sequence(Ego4D_Narration):
    '''
    Building upon Ego4D_Narration, this class is used to generate a sequence of data for each scenario
    The class accept pre-definded Ego4D_Narration to initialize it
    '''
    def __init__(self, parent_obj, num_sequence=5):
        # if isinstance(parent_obj, Ego4D_Narration):
        #     pass
        # else:
        #     parent_obj = parent_obj.dataset
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


def prepare_imu2clip(filter_video_uid):
    annotation_dir = 'ego4d/splits'
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
    print('Scenario count:', scenario_count)
    return imu2clip_label, scenario_map
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

class Ego4D_Understanding(Dataset):
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
    def __getitem__(self, i):
        dict_out = self.window_idx[i].copy()
        print(dict_out)

        audio_label = os.path.join(self.folder, 'audio_label', f"{dict_out['video_uid']}.json")
        audio_label = json.load(open(audio_label, 'r'))
        print(audio_label)

        # scenario_vec = np.zeros(len(self.scenario_map), dtype=float)
        # scenario_vec[dict_out['scenario']] = 1
        # dict_out['scenario'] = scenario_vec
        # if 'imu' in self.modal:
        #     imu = np.load(os.path.join(self.folder, 'processed_imu', f"{uid}.npy")).astype(np.float32)
        #     imu = imu[:, int(w_s*self.sr_imu): int(w_e*self.sr_imu)]
        #     dict_out["imu"] = imu
        # if 'audio' in self.modal:
        #     audio, sr = librosa.load(os.path.join(self.folder, 'audio', f"{uid}.mp3"), 
        #                             offset=w_s, duration=self.window_sec, sr=self.sr_audio)
        #     if audio.shape[-1] < self.sr_audio * self.window_sec:
        #         audio = np.pad(audio, (0, self.sr_audio * self.window_sec - audio.shape[-1]))
        #     dict_out["audio"] = audio
        # if 'capture24' in self.modal:
        #     dict_out['capture24'] = self.capture24[:, i]
        return dict_out
    


