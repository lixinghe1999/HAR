import numpy as np
from torch.utils.data import Dataset
import os
import json
import matplotlib.pyplot as plt
import soundfile as sf
from utils.egoexo_utils import prepare_pose
from utils.text_cluster import label_dict, cluster_plot, cluster_map, close_to
import torchvision

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
    def __init__(self, data_dir='../dataset/egoexo', split='train', window_sec=2, modal=['audio', 'imu',]):
        self.data_dir = data_dir
        self.modal = modal
        self.meta = json.load(open(os.path.join(data_dir, 'takes.json')))
        self.takes_by_uid = {x["take_uid"]: x for x in self.meta}
        print('Total number of takes:', len(self.takes_by_uid))

        self.pose_root = os.path.join(self.data_dir, "annotations/ego_pose", split, "body/annotation")
        self.camera_root = os.path.join(self.data_dir, "annotations/ego_pose", split, "camera_pose")
        self.pose_files = os.listdir(self.pose_root)

        self.annotation_dir = os.path.join(data_dir, 'annotations')
        self.atomic = os.path.join(self.annotation_dir, 'atomic_descriptions_{}.json'.format(split))
        self.atomic = json.load(open(self.atomic))['annotations']
        self.all_descriptions = []
        self.window_sec = window_sec

        for take_uid, xs in self.atomic.items():
            '''make sure the file exist'''
            if take_uid not in self.takes_by_uid:
                continue
            take_meta = self.takes_by_uid[take_uid]
            if take_meta['vrs_relative_path'] == None:
                continue
            take_path = os.path.join(self.data_dir, take_meta['root_dir'], take_meta['vrs_relative_path'])
            if not os.path.exists(take_path):
                continue
            for x in xs:
                if x['rejected']:
                    continue
                descriptions = x['descriptions']
                for description in descriptions:
                    self.all_descriptions.append((take_uid, description))
        print('Total number of atomic descriptions:', len(self.all_descriptions))
    def __len__(self):
        return len(self.all_descriptions)
    def __getitem__(self, idx):
        dict_out = {}
        take_uid, description = self.all_descriptions[idx]
        dict_out['text'] = description['text']

        take_meta = self.takes_by_uid[take_uid]
        dict_out['task_name'] = take_meta['task_name']
        dict_out['parent_task_name'] = take_meta['parent_task_name'] 
        take_path = os.path.join(self.data_dir, take_meta['root_dir'], take_meta['vrs_relative_path'])
        start = description['timestamp'] - self.window_sec/2

        if 'audio' in self.modal:
            audio_path = take_path.replace('.vrs', '.flac')
            audio = sf.read(audio_path, start=int(start*48000), stop=int(start*48000) + int(self.window_sec * 48000))[0].T
            if audio.shape[1] < self.window_sec * 48000:
                audio = np.pad(audio, ((0, 0), (0, int(self.window_sec * 48000 - audio.shape[1]))), 'constant', constant_values=0)
            dict_out['audio'] = audio
        if 'imu' in self.modal:
            imu_path = take_path.replace('.vrs', '.npy')
            imu = np.load(imu_path).astype(np.float32)
            imu = imu[:, int(start * 800): int(start * 800) + int(self.window_sec * 800)]
            if imu.shape[1] < self.window_sec * 800:
                imu = np.pad(imu, ((0, 0), (0, int(self.window_sec * 800 - imu.shape[1]))), 'constant', constant_values=0)
            dict_out['imu'] = imu.T
        if 'video' in self.modal:
            video_dir = os.path.join(os.path.dirname(take_path), 'frame_aligned_videos/downscaled/448/')
            videos = os.listdir(video_dir)
            for video in videos:
                if video.endswith('_214-1.mp4'):
                    video_path = os.path.join(video_dir, video)
                    # load mid-frame
                    images, _, _= torchvision.io.read_video(video_path, start_pts=description['timestamp'], 
                                                       end_pts=description['timestamp'], pts_unit='sec')
                    dict_out['video'] = images
                    break
                    
        return dict_out
class Baseline_Dataset(Dataset):
    '''
    load npy dataset from the folder 'small_dataset'
    [hhar, motion, shoaib, uci]
    '''
    def __init__(self, datasets=['hhar', 'motion', 'shoaib', 'uci'], supervised=False, split='train'):
        datas, labels = [], []
        if supervised:
            assert len(datasets) == 1
        for data_dir in datasets:
            data_dir = os.path.join('small_dataset', data_dir)
            data = np.load(data_dir + '/data_20_120.npy').astype(np.float32)
            arr = np.arange(data.shape[0])
            np.random.shuffle(arr)
            data = data[arr]
            if data.shape[2] > 6:
                data = data[:, :, :6]
            if split == 'train':
                data = data[:int(0.8 * data.shape[0])]
            else:
                data = data[int(0.8 * data.shape[0]):]
            datas.append(data)

            label = np.load(data_dir + '/label_20_120.npy').astype(np.int64)
            label = label[:, 0, 0]
            label = label[arr]
            if split == 'train':
                label = label[:int(0.8 * label.shape[0])]
            else:
                label = label[int(0.8 * label.shape[0]):]
            assert data.shape[0] == label.shape[0]            
            labels.append(label)

        self.data = np.concatenate(datas, axis=0)
        if supervised:
            self.labels = np.concatenate(labels, axis=0)
        self.supervised = supervised
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        if self.supervised:
            return {'imu': self.data[idx], 'label': self.labels[idx]}
        else:
            return {'imu': self.data[idx], 'label': None}

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
    from tqdm import tqdm
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
