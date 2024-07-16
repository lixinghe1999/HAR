import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa import MUSIC
from .multi_channel import fs, nfft, mic_center, mic_array
import librosa

def init_music():
    kwargs = {'L': mic_center + mic_array,
            'fs': fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(360)),
            'colatitude': np.deg2rad(np.arange(180)),
            'num_src': 1
    }
    algo = MUSIC(**kwargs)
    return algo

def inference_music(algo, audio,):
    predictions = []
    intervals = librosa.effects.split(y=audio, top_db=20, ref=1)
    n_windows = np.shape(intervals)[0]
    for i in range(n_windows):
        start = intervals[i][0]
        end = intervals[i][1]
        if end - start < nfft:
            continue
        data = audio[:, start:end]
        # detect voice activity
        stft_signals = stft(data, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
        algo.locate_sources(stft_signals)
        predictions.append(np.rad2deg(algo.azimuth_recon[0]))
    predictions = np.array(predictions)
    return predictions