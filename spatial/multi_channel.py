import numpy as np
from aoa import init_music, inference_music
from range import calculate_range
import os
import matplotlib.pyplot as plt
import librosa
fs = 16000 
nfft = 1024
max_order = 10
snr_lb, snr_ub = 0, 30
mic_center = np.c_[[2, 2, 1.7]]
# mic_array= np.c_[[ 0.06,  0.0, 0.0], [ -0.06,  0.0, 0.0],
#                  [ 0,  0.06, 0.0], [ 0,  -0.06, 0.0],]

mic_array_aria = np.c_[[ 0.03546091, -0.09185805, -0.07152638], [-0.00630641, -0.054662, -0.03122935],
                    [ 0.03252111, -0.01865076, -0.01542986], [-0.00923072, -0.00177898, -0.00708757],
                    [-0.0068401,  -0.10160788, -0.07526885], [-0.00162908,  0.06371858, -0.08164837],
                    [ 0.00298007, -0.05102968, -0.16957706]]

mic_array_seeed = np.c_[             
                    [ -0.03,  0.06, 0.0],
                    [ 0.03,  0.06, 0.0],
                    [ 0.06,  0.0, 0.0],
                    [ 0.03,  -0.06, 0.0],
                    [ -0.03,  -0.06, 0.0],
                    [ -0.06,  0, 0.0], 
                    ]

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

if __name__ == '__main__':
    audio_names = os.listdir('dataset')
    audio_names = [os.path.join('dataset', audio_name) for audio_name in audio_names]
    
    # audio_names = ['dataset/20240731_154148_micarray.wav']
    for audio_name in audio_names:
        audio, sr = librosa.load(audio_name, sr=fs, mono=False)
        audio = audio[[0, 1, 2, 3, 4, 5], :] 

        algo = init_music(mic_center, mic_array_seeed, fs, nfft)
        predictions = inference_music(algo, audio)
        print(predictions)

        estimated_range = calculate_range(audio, mic_center, mic_array_seeed, fs)
        print(estimated_range)