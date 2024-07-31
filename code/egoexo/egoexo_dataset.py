import numpy as np
from torch.utils.data import Dataset
import os
import json
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from .egoexo_utils import prepare_pose
import torchvision
from tqdm import tqdm

class EgoExo_pose(Dataset):
    def __init__(self, data_dir='../dataset/egoexo', split='train', window_sec=1, max_frames=1000000, stride=5):
        self.data_dir = data_dir
        self.meta = json.load(open(os.path.join(data_dir, 'takes.json')))
        self.takes_by_uid = {x["take_uid"]: x for x in self.meta}

        self.window_sec = window_sec
        self.slice_window = int(self.window_sec * 30  / 3) 
        self.joint_names = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']

        self.pose_root = os.path.join(self.data_dir, "annotations/ego_pose", split, "body/annotation")
        self.camera_root = os.path.join(self.data_dir, "annotations/ego_pose", split, "camera_pose")
        self.annotation_dir = os.path.join(data_dir, 'annotations')
        self.atomic = os.path.join(self.annotation_dir, 'atomic_descriptions_{}.json'.format(split))
        self.atomic = json.load(open(self.atomic))['annotations']

        self.pose_files = os.listdir(self.pose_root)
        self.camera_files = os.listdir(self.camera_root)
        self.pose, self.aria_trajectory, self.pose_sum = prepare_pose(data_dir, split, self.takes_by_uid, slice_window=self.slice_window, max_frames=max_frames)
        self.stride = stride
        print('Total number of poses frames:', self.__len__())
        self.poses_takes_uids = list(self.pose.keys())
    def __len__(self):
        return sum(self.pose_sum) // self.stride
    def __getitem__(self, idx):
        take_id = 0
        idx = idx * self.stride
        while True:
            if idx < self.pose_sum[take_id]:
                break
            idx -= self.pose_sum[take_id]
            take_id += 1
        take_uid = self.poses_takes_uids[take_id]
        pose = self.pose[take_uid]

        capture_frames = list(pose.keys())
        frames_idx = idx + self.slice_window
        frames_window = capture_frames[frames_idx - self.slice_window:frames_idx]

        skeletons_window = []
        flags_window = []
        aria_window = []
        for frame in frames_window:
            skeleton = pose[frame][0]['annotation3D']
            skeleton, flags = parse_skeleton(skeleton, self.joint_names)
            aria_trajectory = self.aria_trajectory[take_uid][frame]
            skeletons_window.append(skeleton)
            flags_window.append(flags)
            aria_window.append(aria_trajectory)
        dict_out = {}
        dict_out['gt'] = np.array(skeletons_window, dtype=np.float32)
        dict_out['visible'] = np.array(flags_window, dtype=np.float32)
        dict_out['camera_pose'] = np.array(aria_window, dtype=np.float32)

        timestamp = int(frames_window[-1]) / 30 - self.slice_window // 2
        take_meta = self.takes_by_uid[take_uid]
        print(take_meta['root_dir'], take_meta['vrs_relative_path'])
        take_path = os.path.join(self.data_dir, take_meta['root_dir'], take_meta['vrs_relative_path'])
        audio_path = take_path.replace('.vrs', '.flac')
        imu_path = take_path.replace('.vrs', '.npy')
        features_path = os.path.join(self.data_dir, take_meta['root_dir'], 'features.npy')
        tags_path = os.path.join(self.data_dir, take_meta['root_dir'], 'tags.json')
        music_path = os.path.join(self.data_dir, take_meta['root_dir'], 'music.npy')

        audio, imu, features, tags, music = load_data(timestamp, audio_path, imu_path, features_path, tags_path, music_path, self.window_sec)
        dict_out['audio'] = audio
        dict_out['imu'] = imu
        dict_out['features'] = features
        dict_out['tags'] = tags
        dict_out['ssl'] = music
        return dict_out

def parse_skeleton(skeleton, joint_names):
    poses = []
    flags = []
    keypoints = skeleton.keys()
    for keyp in joint_names:
        if keyp in keypoints:
            flags.append(1) #visible
            poses.append([skeleton[keyp]['x'], skeleton[keyp]['y'], skeleton[keyp]['z']]) #visible
        else:
            flags.append(0) #not visible
            poses.append([-1,-1,-1]) #not visible
    return poses, flags

class EgoExo_atomic(Dataset):
    def __init__(self, pre_compute_json=None, folder='../dataset/egoexo/', 
                 split='train', window_sec=2, modal=['audio', 'imu',]):
        '''
        pre_compute_json: str, path to pre-computed json file (by self.save_json)
        folder: str, path to the dataset directory
        split: str, 'train' or 'val'
        window_sec: int, window size in seconds
        modal: list, ['audio', 'imu', 'video']
        '''
        self.folder = folder
        self.modal = modal
        self.meta = json.load(open(os.path.join(folder, 'takes.json')))
        self.takes_by_uid = {x["take_uid"]: x for x in self.meta}
        self.window_sec = window_sec
        if pre_compute_json is not None:
            self.window_idx = json.load(open(pre_compute_json))
        else:
            self.annotation_dir = os.path.join(folder, 'annotations')
            self.atomic = os.path.join(self.annotation_dir, 'atomic_descriptions_{}.json'.format(split))
            self.atomic = json.load(open(self.atomic))['annotations']
            self.window_idx = []
            for take_uid, xs in self.atomic.items():
                '''make sure the file exist'''
                if take_uid not in self.takes_by_uid:
                    continue
                take_meta = self.takes_by_uid[take_uid]
                if take_meta['vrs_relative_path'] == None:
                    continue
                take_path = os.path.join(self.folder, take_meta['root_dir'], take_meta['vrs_relative_path']
                                         .replace('.vrs', '.flac'))
                if not os.path.exists(take_path):
                    continue
                for x in xs:
                    if x['rejected']:
                        continue
                    descriptions = x['descriptions']
                    for description in descriptions:
                        description['take_uid'] = take_uid
                        description['scenario'] = self.takes_by_uid[take_uid]['task_name']
                        # description['parent_scenario'] = self.takes_by_uid[take_uid]['parent_task_name']
                        del description['ego_visible']
                        del description['best_exo']
                        del description['unsure'] 
                        self.window_idx.append(description)
        self.sr_imu = 200
        self.channel_imu = 6
        self.sr_audio = 16000
        print('Total number of atomic descriptions:', len(self.window_idx))
    def __len__(self):
        return len(self.window_idx)
    def split_with_scenario(self, ratio=0.8):
        train_idx = []
        test_idx = []
        scenario_idx = {}
        for i, data in enumerate(self.window_idx):
            scenario = data['scenario']
            if scenario not in scenario_idx:
                scenario_idx[scenario] = []
            scenario_idx[scenario].append(i)
            self.window_idx[i]['scenario'] = list(scenario_idx.keys()).index(scenario)
        
        for scenario, idx in scenario_idx.items():
            train_size = int(len(idx) * ratio)
            train_idx += idx[:train_size]
            test_idx += idx[train_size:]
        print('Number of scenarios:', len(scenario_idx))
        print('Total number of train:', len(train_idx), 'Total number of test:', len(test_idx))
        return train_idx, test_idx
    def negative(self):
        new_all_descriptions = []
        gap_descriptions = []
        for i in tqdm(range(len(self.window_idx) - 1)):
            take_uid1 = self.window_idx[i]['take_uid']
            time_stamp1 = self.window_idx[i]['timestamp']
            if take_uid1 == self.window_idx[i + 1]['take_uid']: # still on the same take
                time_stamp2 = self.window_idx[i + 1]['timestamp']
                new_descrption = self.window_idx[i].copy()
                new_descrption['text'] = 'unsure'
                del new_descrption['sound']
                gap_descriptions.append(time_stamp2 - time_stamp1)
                if time_stamp2 - time_stamp1 < 2 * self.window_sec:
                    continue
                else:
                    for timestamp in np.arange(time_stamp1 + self.window_sec, time_stamp2 - self.window_sec, self.window_sec):
                        new_descrption['timestamp'] = float(timestamp)
                        new_all_descriptions.append(new_descrption)
        self.window_idx = new_all_descriptions
        print('Total number of Negative samples', len(self.window_idx))
        print('Average gap between two descriptions:', np.mean(gap_descriptions))
    def prune_slience(self, fname='resources/egoexo_atomic_prune.json'):
        new_window_idx = []
        pruned, kept = 0, 0
        for i in tqdm(range(0, self.__len__())):
            data = self.__getitem__(i)
            audio = data['audio']
            valid_audio = librosa.effects.split(y=audio, top_db=20, ref=1)
            if len(valid_audio) == 0: # no sound
                pruned += 1
            else:
                new_window_idx.append(self.window_idx[i])
                kept += 1
        print('Pruned:', pruned, 'Kept:', kept)
        self.window_idx = new_window_idx
        self.save_json(fname)
    def save_json(self, fname):
        json.dump(self.window_idx, open(fname, 'w'), indent=4)
    def add(self, idx, key, value):
        self.window_idx[idx][key] = value
    def __getitem__(self, idx):
        dict_out = self.window_idx[idx].copy()
        take_meta = self.takes_by_uid[dict_out['take_uid']]
        dict_out['root_dir'] = take_meta['root_dir']

        dict_out['task_name'] = take_meta['task_name']
        dict_out['parent_task_name'] = take_meta['parent_task_name'] 

        take_path = os.path.join(self.folder, take_meta['root_dir'], take_meta['vrs_relative_path'][:-4])
        start = dict_out['timestamp'] - self.window_sec/2
        if 'audio' in self.modal:
            dict_out['audio_path'] = take_path + '.flac'
            start_frame = int(start * 48000)
            stop_frame = start_frame + int(self.window_sec * 48000)
            audio = sf.read(dict_out['audio_path'], start=start_frame, stop=stop_frame, dtype='float32', always_2d=True)[0].T
            audio = audio[:, ::3]
            if audio.shape[-1] < self.window_sec * self.sr_audio:
                audio = np.pad(audio, ((0, 0), (0, int(self.window_sec * self.sr_audio - audio.shape[-1]))), 'constant', constant_values=0)
            if 'spatial_audio' in self.modal:
                dict_out['spatial_audio'] = audio
            # audio = librosa.load(dict_out['audio_path'], sr=self.sr_audio, mono=True, offset=start, duration=self.window_sec)[0]
            dict_out['audio'] = audio[0] # MONO
        
        if 'imu' in self.modal:
            down_sample_factor = 800 // self.sr_imu
            dict_out['imu_path'] = take_path + '.npy'
            imu = np.load(dict_out['imu_path']).astype(np.float32)[:, ::down_sample_factor]
            imu = imu[:self.channel_imu, int(start * self.sr_imu): int(start * self.sr_imu) + int(self.window_sec * self.sr_imu)]
            if imu.shape[1] < self.window_sec * self.sr_imu:
                imu = np.pad(imu, ((0, 0), (0, int(self.window_sec * self.sr_imu - imu.shape[1]))), 'constant', constant_values=0)
            dict_out['imu'] = imu
        if 'image' in self.modal:
            video_dir = os.path.join(os.path.dirname(take_path), 'frame_aligned_videos/downscaled/448/')
            videos = os.listdir(video_dir)
            for video in videos:
                if video.endswith('_214-1.mp4'):
                    dict_out['video_path'] = os.path.join(video_dir, video)
                    images, _, _= torchvision.io.read_video(dict_out['video_path'], 
                                                            start_pts=dict_out['timestamp'], end_pts=dict_out['timestamp'], pts_unit='sec')
                    dict_out['image'] = images
                    break           
        return dict_out


def visualize_audio(audio, folder):
    plt.figure()
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(f"Microphone signal")
    plt.plot(audio.T)
    plt.savefig(folder + '/audio.png')
    sf.write(folder + '/audio.wav', audio.T, 48000)
def visualize_pose(pose, visible, folder):
    assert pose.shape[1] == 17
    '''visualize body keypoints in 3D'''
    from matplotlib.animation import FuncAnimation  
    def animate(i):
        p = pose[i]
        v = visible[i]
        p[v == 0] = 0
        points_line._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        
        for connection, line in zip(connections, lines):
            if v[connection[0]] == 0 or v[connection[1]] == 0:
                continue
            line.set_data_3d([p[connection[0], 0], p[connection[1], 0]],
                            [p[connection[0], 1], p[connection[1], 1]],
                            [p[connection[0], 2], p[connection[1], 2]])
        # axs.view_init(elev=20, azim=3*i)
        return points_line,
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.scatter(0, 0, 0, c='r', marker='o')
    points_line = axs.scatter(pose[0, :, 0], pose[0, :, 1], pose[0, :, 2], c='b', marker='o')
    # x_min, x_max = np.min(pose[:, :, 0]), np.max(pose[:, :, 0])
    # y_min, y_max = np.min(pose[:, :, 1]), np.max(pose[:, :, 1])
    # # print(x_min, x_max, y_min, y_max)
    # axs.set_xlim(x_min, x_max)
    # axs.set_ylim(y_min, y_max)
    connections = [(0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8),
                (8, 10), (6, 12), (5, 11), (11, 12),
                (12, 14), (11, 13), (14, 16), (13, 15)]
    lines = []
    for _ in connections:
        lines.append(axs.plot([0, 0], [0, 0], [0, 0], c='b')[0])
    anim = FuncAnimation(fig, animate, frames = pose.shape[0], interval = 100, blit = True)
    anim.save(folder + '/pose_animation.gif', writer = 'ffmpeg', fps = 10)     
def visualize_imu(imu, folder):
    fig = plt.figure()
    assert imu.shape[0] == 12
    imu = imu - np.mean(imu, axis=1, keepdims=True)
    left, right = imu[:6], imu[6:]
    axs = fig.add_subplot(121,)
    axs.plot(left.T, label=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    plt.legend()
    axs = fig.add_subplot(122,)
    axs.plot(right.T, label=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    plt.legend()
    plt.savefig(folder + '/imu.png')
def overlap_pose_atomic():
    annotation_dir = '../dataset/egoexo/annotations'
    split = 'train'
    atomic = os.path.join(annotation_dir, 'atomic_descriptions_{}.json'.format(split))
    atomic_files = json.load(open(atomic))['annotations']

    pose = os.path.join(annotation_dir, 'ego_pose', split, 'body/annotation')
    pose_files = os.listdir(pose)
    
    both_files = []
    for atomic_file in atomic_files:
        if atomic_file + '.json' in pose_files:
            both_files.append(atomic_file)
    print('Total number of atomic descriptions:', len(atomic_files), 'Total number of poses:', len(pose_files), 'Total number of both:', len(both_files))
if __name__ == '__main__':
    # overlap_pose_atomic()
    # dataset = EgoExo_pose(split='train')
    # idx = random.randint(0, len(dataset)-1)
    # # idx = 46
    # print(len(dataset))
    # data = dataset[idx]
    # print(idx, data['gt'].shape, data['visible'].shape, data['camera_pose'].shape, 
    #       data['audio'].shape, data['imu'].shape)
    # folder = 'figs'
    # visualize_pose(data['gt'], data['visible'], folder)
    # visualize_audio(data['audio'], folder)
    # visualize_imu(data['imu'], folder)

    # for idx, data in enumerate(tqdm(dataset)):
    #     print(data['imu'].shape, data['audio'].shape)
    #     pass

    # dataset = EgoExo_pose(split='train')
    dataset = EgoExo_atomic(window_sec=2, modal=['efficientAT'])
    for idx, data in enumerate(tqdm(dataset)):
        # for key, value in data.items():
        #     if type(value) == np.ndarray:
        #         print(key, value.shape)
        #     else:
        #         print(key, value)
        # folder = 'figs'
        # visualize_audio(data['audio'], folder)
        # visualize_imu(data['imu'], folder)
        break
        # pass
