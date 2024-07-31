from ego4d.ego4d_dataset import Ego4D_Narration
import pandas as pd
import math
import string
from .extract_imu import get_ego4d_metadata
import os
from tqdm import tqdm
import json

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
        for idx, row in tqdm(ego4d_sound_meta.iterrows()):
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
        print(f"Total {len(self.window_idx)} windows")
        self.scenario_map = {v:k for k,v in self.scenario_map.items()}
        print('Total scenarios:', len(self.scenario_map))
        self.sr_audio = 16000
        self.sr_imu = 200
