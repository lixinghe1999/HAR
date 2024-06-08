'''
Prepare all type of annotations here
'''
import os
import json
from tqdm import tqdm
import numpy as np
from projectaria_tools.core import data_provider
import scipy.signal as signal
import soundfile as sf

def translate_pose(pose_data, camera_data, coord='global'): 
    trajectory = {}
    for key in camera_data.keys():
        if "aria" in key:
            aria_key =  key
            break
    first = next(iter(pose_data))
    first_cam = camera_data[aria_key]['camera_extrinsics'][first]
    T_first_camera = np.eye(4)
    T_first_camera[:3, :] = np.array(first_cam)

    for frame in pose_data:
        current_anno = pose_data[frame]
        current_cam = camera_data[aria_key]['camera_extrinsics'][frame]
        T_world_camera_ = np.eye(4)
        T_world_camera_[:3, :] = np.array(current_cam)

        if coord == 'global':
            T_world_camera = np.linalg.inv(T_world_camera_)
        elif coord == 'aria':
            T_world_camera = np.dot(T_first_camera,np.linalg.inv(T_world_camera_))
        else:
            T_world_camera = T_world_camera_

        for idx in range(len(current_anno)):
            joints = current_anno[idx]['annotation3D']
            for joint_name in joints:
                joint4d = np.ones(4)
                joint4d[:3] = np.array([joints[joint_name]["x"], joints[joint_name]["y"], joints[joint_name]["z"]])

                if coord == 'global':
                    new_joint4d = joint4d
                elif coord == 'aria':
                    new_joint4d = T_first_camera.dot(joint4d)
                else:
                    new_joint4d = T_world_camera_.dot(joint4d) 

                joints[joint_name] = {}
                joints[joint_name]["x"] = new_joint4d[0]
                joints[joint_name]["y"] = new_joint4d[1]
                joints[joint_name]["z"] = new_joint4d[2]
            current_anno[idx]["annotation3D"] = joints
        traj = T_world_camera[:3,3]
        trajectory[frame] = traj
        pose_data[frame] = current_anno
    return pose_data, trajectory
def prepare_pose(data_dir, split, takes_by_uid, slice_window=1, max_frames=1000):
    pose_root = os.path.join(data_dir, "annotations/ego_pose", split, "body/annotation")
    camera_root = os.path.join(data_dir, "annotations/ego_pose", split, "camera_pose")
    annotation_dir = os.path.join(data_dir, 'annotations')
    atomic = os.path.join(annotation_dir, 'atomic_descriptions_{}.json'.format(split))
    atomic = json.load(open(atomic))['annotations']

    pose_files = os.listdir(pose_root)
    print('Total number of poses:', len(pose_files))
    pose = {}
    aria_trajectory = {}
    pose_sum = []
    for pose_file in tqdm(pose_files):
        camera_pose = os.path.join(camera_root, pose_file)
        if not os.path.exists(camera_pose):
            continue
        pose_data = json.load(open(os.path.join(pose_root, pose_file)))
        camera_data = json.load(open(camera_pose))
        pose_data, camera_data = translate_pose(pose_data, camera_data)
        take_id = pose_file[:-5]
        
        pose[take_id] = pose_data
        aria_trajectory[take_id] = camera_data
        pose_sum.append(len(pose_data) - slice_window)
        # break
        if sum(pose_sum) > max_frames: # don't use the full dataset
            # print('early stop, do not use full dataset')
            break
    return pose, aria_trajectory, pose_sum
def read_audio_all(provider, stream_id_mic, sample_rate=48000):
    # audio = [[] for c in range(0, 7)]
    audio = []
    for index in range(0, provider.get_num_data(stream_id_mic)):
        audio_data_i = provider.get_audio_data_by_index(stream_id_mic, index)
        audio_signal_block = audio_data_i[0].data
        time_stamp_block = audio_data_i[1].capture_timestamps_ns
        N_channel = len(audio_signal_block) // len(time_stamp_block)
        audio_signal_block = np.array(audio_signal_block).reshape(-1, N_channel)
        audio.append(audio_signal_block)
        # for c in range(0, 7):
        #     audio[c] += audio_signal_block[c::7]
    # print(N_channel)
    # audio = np.array(audio)
    audio = np.concatenate(audio, axis=0)
    audio = np.round(audio / (2**31 - 1), 8)
    # audio = audio / np.max(np.abs(audio), axis=0)
    rms = np.sqrt(np.mean(audio**2, axis=0, keepdims=True))
    norm_factor = 10**(-25/20) / rms  # -25 dBFS target
    audio = audio * norm_factor
    return audio
def read_imu_all(provider, stream_id_imu, sample_rate=800):
    imu = []
    data_last = provider.get_imu_data_by_index(stream_id_imu, provider.get_num_data(stream_id_imu)-1)
    data_first = provider.get_imu_data_by_index(stream_id_imu, 0)
    time_shift = (data_last.capture_timestamp_ns - data_first.capture_timestamp_ns) * 1e-9
    for index in range(0, provider.get_num_data(stream_id_imu)):
        imu_data = provider.get_imu_data_by_index(stream_id_imu, index)
        imu_data = [imu_data.accel_msec2[0], imu_data.accel_msec2[1], imu_data.accel_msec2[2], imu_data.gyro_radsec[0], 
                    imu_data.gyro_radsec[1], imu_data.gyro_radsec[2],] #imu_data.capture_timestamp_ns * 1e-9]
        imu.append(imu_data)
    imu = np.stack(imu, axis=1)
    imu = signal.resample(imu, int(sample_rate * time_shift), axis=1)
    return imu
def extract(path):
    provider = data_provider.create_vrs_data_provider(path)
    stream_id_mic = provider.get_stream_id_from_label("mic")
    stream_id_imu_right = provider.get_stream_id_from_label("imu-right")
    stream_id_imu_left = provider.get_stream_id_from_label("imu-left")
    audio = read_audio_all(provider, stream_id_mic)
    # audio = audio[:, 0] # mono only
    sf.write(path.replace('.vrs', '.flac'), audio, 48000)
    # imu_right = read_imu_all(provider, stream_id_imu_right)
    # imu_left = read_imu_all(provider, stream_id_imu_left)
    # if imu_right.shape[1] > imu_left.shape[1]:
    #     imu_right = imu_right[:, :imu_left.shape[1]]
    # else:
    #     imu_left = imu_left[:, :imu_right.shape[1]]
    # imu = np.concatenate([imu_right, imu_left], axis=0)
    # np.save(path.replace('.vrs', '.npy'), imu)  
def extract_data_vrs():
    import multiprocessing
    data_dir = '../dataset/egoexo/takes'
    paths = []
    for take_folder in os.listdir(data_dir):
        for take_file in os.listdir(os.path.join(data_dir, take_folder)):
            if not take_file.endswith('.vrs'):
                continue
            path = os.path.join(data_dir, take_folder, take_file)
            paths.append(path)
        # break
    with multiprocessing.Pool(8) as p:
      r = list(tqdm(p.imap(extract, paths), total=len(paths)))
if __name__ == "__main__":
    extract_data_vrs()