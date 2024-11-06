import torch.utils.data as data
import os
import librosa
import numpy as np
class CustomDataset(data.Dataset):
    def __init__(self, folder, length=5):
        self.folder = folder
        files = os.listdir(folder)
        audios = [file for file in files if file.endswith('.mp3')]
        imus = [file for file in files if file.endswith('.npy')]
        
        self.data = []
        class_maps = {}
        for audio in audios:
            idx, class_name, _ = audio.split(',')
            print(idx, class_name)
            if class_name not in class_maps:
                class_maps[class_name] = len(class_maps)
            duration = librosa.get_duration(path=os.path.join(folder, audio))
            if duration < length:
                self.data.append([audio, (0, duration)])
            else:
                for i in range(int(duration//length)):
                    self.data.append([audio, (i*length, (i+1)*length), class_maps[class_name]])
        print("Data: ", len(self.data))

    def __getitem__(self, index):
        file, (start, end), class_idx = self.data[index]
        audio_file = os.path.join(self.folder, file)
        audio, sr = librosa.load(audio_file, sr=16000, offset=start, duration=end-start)
        imu = np.zeros((6, 1000), dtype=np.float32)
        return {'audio': audio, 'imu': imu}, class_idx
    
    def __split__(self, ratio=0.8):
        # split the dataset into train and test by each class
        class_idxs = {}
        for i in range(len(self.data)):
            _, _, class_idx = self.data[i]
            if class_idx not in class_idxs:
                class_idxs[class_idx] = []
            class_idxs[class_idx].append(i)
        train_data, test_data = [], []
        for class_idx in class_idxs:
            idxs = class_idxs[class_idx]
            split_idx = int(len(idxs) * ratio)
            train_data += idxs[:split_idx]
            test_data += idxs[split_idx:]
        return train_data, test_data
    def __len__(self):
        return len(self.data)
    
if __name__ == '__main__':
    dataset = CustomDataset('../dataset/aiot/Lixing_home-20241106_082431_132')
    train_data, test_data = dataset.__split__()
    train_dataset = data.Subset(dataset, train_data)
    test_dataset = data.Subset(dataset, test_data)

