import numpy as np
import librosa
import os

fs = 16000 
nfft = 1024
max_order = 10
snr_lb, snr_ub = 0, 30
mic_center = np.c_[[2, 2, 1.7]]
# mic_array= np.c_[[ 0.06,  0.0, 0.0], [ -0.06,  0.0, 0.0],
#                  [ 0,  0.06, 0.0], [ 0,  -0.06, 0.0],]

mic_array = np.c_[[ 0.03546091, -0.09185805, -0.07152638], [-0.00630641, -0.054662, -0.03122935],
                    [ 0.03252111, -0.01865076, -0.01542986], [-0.00923072, -0.00177898, -0.00708757],
                    [-0.0068401,  -0.10160788, -0.07526885], [-0.00162908,  0.06371858, -0.08164837],
                    [ 0.00298007, -0.05102968, -0.16957706]]
from spafe.features.gfcc import gfcc

def gtcc(audio):
    gfccs = gfcc(audio, fs=fs, num_ceps=36, nfilts=48, nfft=1024).astype(np.float32)
    return gfccs[np.newaxis, :]
def gccphat(audio1, audio2, gcc_phat_len):
    n = audio1.shape[0] + audio2.shape[0] 
    X = np.fft.rfft(audio1, n=n)
    Y = np.fft.rfft(audio2, n=n)
    R = X * np.conj(Y)
    cc = np.fft.irfft(R / (1e-6 + np.abs(R)),  n=(1 * n))
    cc = np.concatenate((cc[-gcc_phat_len:], cc[: gcc_phat_len+1])).astype(np.float32)
    return cc
def mel_gccphat(self, audio1, audio2):
    n = audio1.shape[0] + audio2.shape[0]
    melfb = librosa.filters.mel(sr=self.sr, n_fft=n, n_mels=40)
    X = np.fft.rfft(audio1, n=n)
    Y = np.fft.rfft(audio2, n=n)
    R = X * np.conj(Y)
    R_mel = melfb * R
    cc = np.fft.irfft(R_mel / (np.abs(R_mel) + 1e-6), n=n)
    cc = np.concatenate((cc[:, -self.gcc_phat_len:], cc[:, :self.gcc_phat_len+1]), axis=1).astype(np.float32)
    cc = np.expand_dims(cc, axis=0)
    return cc
def output_all():
    import multiprocessing
    from tqdm import tqdm
    data_dir = '../dataset/egoexo/takes'
    paths = []
    for take_folder in os.listdir(data_dir):
        for take_file in os.listdir(os.path.join(data_dir, take_folder)):
            if not take_file.endswith('.flac'):
                continue
            path = os.path.join(data_dir, take_folder, take_file)
            paths.append(path)
    paths = paths[:1]
    with multiprocessing.Pool(12) as p:
      r = list(tqdm(p.imap(MUSIC_func, paths), total=len(paths)))
    # with multiprocessing.Pool(12) as p:
    #     r = list(tqdm(p.imap(pair_representation, paths), total=len(paths)))
    # with multiprocessing.Pool(12) as p:
    #     r = list(tqdm(p.imap(each_representation, paths), total=len(paths)))
def each_representation(file_path, hop_frame=int(fs * 1), func=gtcc, name='gtcc'):
    audio, _ = librosa.load(file_path, sr=fs, mono=False)
    n_windows = int(np.ceil(audio.shape[1] / hop_frame))
    save_data = []
    for i in range(n_windows):
        start = i * hop_frame
        end = min((i+1) * hop_frame, audio.shape[1])
        data = audio[:, start:end]
        data = np.pad(data, ((0, 0), (0, hop_frame - data.shape[1])), mode='constant')
        representations = []
        for i in range(audio.shape[0]):
            representation = func(data[i])
            representations.append(representation)
        representations = np.array(representations)
        save_data.append(representations)
    save_data = np.array(save_data)
    folder = os.path.dirname(file_path)
    np.save(os.path.join(folder, '{}.npy'.format(name)), save_data)
def pair_representation(file_path, hop_frame=int(fs * 1), func=gccphat, name='gccphat'):
    audio, _ = librosa.load(file_path, sr=fs, mono=False)
    n_windows = int(np.ceil(audio.shape[1] / hop_frame))
    save_data = []
    for i in range(n_windows):
        start = i * hop_frame
        end = min((i+1) * hop_frame, audio.shape[1])
        data = audio[:, start:end]
        representations = []
        for i in range(audio.shape[0]):
            for j in range(i+1, audio.shape[0]):
                audio1, audio2 = data[i], data[j] 
                representation = func(audio1, audio2)
                representations.append(representation)
        representations = np.array(representations)
        save_data.append(representations)
    save_data = np.array(save_data)
    folder = os.path.dirname(file_path)
    np.save(os.path.join(folder, '{}.npy'.format(name)), save_data)

if __name__ == "__main__":
    output_all()

   