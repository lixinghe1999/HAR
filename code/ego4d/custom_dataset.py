from torch.utils.data import Dataset
import pandas as pd
import os
import wave
import librosa
# without warning
import warnings
warnings.filterwarnings("ignore")
import datetime

class AiotDataset(Dataset):
    def __init__(self, dataset_dir,):
        self.dataset_dir = dataset_dir
        data = os.listdir(dataset_dir)
        pcm_audio = [os.path.join(dataset_dir, f) for f in data if f.endswith('.m4a')]
        imu = [f for f in data if f.endswith('.csv')][0]
        self.imu = pd.read_csv(os.path.join(dataset_dir, imu), skiprows=10).to_numpy()
        imu_start_time = '2024-10-03 10:52:00'
        self.imu_start_time = datetime.datetime.strptime(imu_start_time, '%Y-%m-%d %H:%M:%S')
        self.imu_sr = 100
        self.audio_sr = 16000
        # self.__compress__(pcm_audio)
        wav_audio = [f.replace('.pcm', '.wav') for f in pcm_audio]
        self.__crop__(wav_audio)

    def __compress__(self, audio_paths):
        for audio_path in audio_paths:
            with open(audio_path, 'rb') as pcmfile:
                pcmdata = pcmfile.read()
                with wave.open(audio_path.replace('.pcm', '.wav'), 'wb') as wavfile:
                    wavfile.setparams((2, 2, 48000, 0, 'NONE', 'NONE'))
                    wavfile.writeframes(pcmdata)

    def __crop__(self, audio_paths, duration=5):
        self.segments = []
        for audio_path in audio_paths:
            audio_duration = librosa.get_duration(path=audio_path)

            basename = os.path.basename(audio_path)[:-4]
            start_time, info = basename.split('_')
            time_shift = (datetime.datetime.strptime(start_time, '%Y-%m-%d %H.%M.%S') - self.imu_start_time).total_seconds()
            for i in range(0, int(audio_duration), duration):
                start = i
                end = i + duration
                self.segments.append([audio_path, start, end, time_shift])
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        audio_path, start, end, time_shift = segment
        audio, sr = librosa.load(audio_path, sr=self.audio_sr, offset=start, duration=end-start)

        start_imu = int((start + time_shift) * self.imu_sr) 
        end_imu = int((end + time_shift) * self.imu_sr)
        imu = self.imu[start_imu:end_imu]
        return audio, imu

    
if __name__ == '__main__':
    from tqdm import tqdm
    dataset = AiotDataset('../dataset/aiot/20241003_har',)
    print(len(dataset))
    for i in tqdm(range(len(dataset))):
        audio, imu = dataset[i]
        print(audio.shape, imu.shape)
