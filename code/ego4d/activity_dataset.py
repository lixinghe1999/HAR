'''
To simplify the problem, Ego4D can be converted to a single-label classification problem.
'''

import numpy as np
from torch.utils.data import Dataset
import os
import soundfile as sf

class Ego4D_Activity(Dataset):
    def __init__(self, folder, window_sec=10, modal=None):
        super().__init__()
        self.folder = folder
        self.window_sec = window_sec
        self.modal = modal if modal else ['audio', 'imu', 'activity']

        self.data = []
        for i, _class in enumerate(os.listdir(folder)):
            class_folder = os.path.join(folder, _class)
            files = os.listdir(class_folder)
            files.sort()  # Ensure consistent order
            audio_files = [f for f in files if f.endswith('.wav') and 'audio' in self.modal]
            imu_files = [f for f in files if f.endswith('.npy') and 'imu' in self.modal]
            for audio, imu in zip(audio_files, imu_files):
                audio_path = os.path.join(class_folder, audio)
                imu_path = os.path.join(class_folder, imu)
                assert audio.split('.')[0] == imu.split('.')[0], f"Mismatch between audio and imu files: {audio}, {imu}"
                scenario = audio.split('_')[0]
                self.data.append((audio_path, imu_path, scenario, i, _class))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_path, imu_path, scenario, activity_index, activity = sample
        audio = sf.read(audio_path, dtype='float32')[0]  # Read audio file
        imu = np.load(imu_path).astype(np.float32)

        return {
            'audio': audio,
            'imu': imu,
            'scenario': scenario,
            'activity': activity    
        }, int(activity_index)
if __name__ == "__main__":
    dataset = Ego4D_Activity(folder='../dataset/ego4d/positive/', window_sec=10, modal=['audio', 'imu', 'activity'])
    
    for i in range(len(dataset)):
        data = dataset[i]
        audio, imu, activity, scenario = data['audio'], data['imu'], data['activity'], data['scenario']
        print(audio.shape, imu.shape, activity, scenario)
        break  # Remove this line to iterate through all samples