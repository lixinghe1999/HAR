'''
organize the MUSIC DOA algorithm
'''

import numpy as np
from scipy.signal import stft
from pyroomacoustics.doa import MUSIC, SRP, TOPS
import librosa
import matplotlib.pyplot as plt

def init(mic_array, fs, nfft, mic_center=0):
    kwargs = {'L': mic_center + mic_array,
            'fs': fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(180)),
            'num_src': 1
    }
    # algo = MUSIC(**kwargs)
    algo = SRP(**kwargs)
    # algo = TOPS(**kwargs)
    return algo

def pra_doa(audio, mic_array, fs, nfft, intervals=None, plot=False):
    algo = init(mic_array, fs, nfft, mic_center=0)
    nfft = algo.nfft
    fs = algo.fs
    predictions = []
    if intervals is None:
        intervals = librosa.effects.split(y=audio, top_db=35, ref=1)
        # remove too short intervals
        intervals = [interval for interval in intervals if interval[1] - interval[0] > nfft]
    n_windows = np.shape(intervals)[0]
    for i in range(n_windows):
        start = intervals[i][0]
        end = intervals[i][1]
        data = audio[:, start:end]
        # detect voice activity
        stft_signals = stft(data, fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
        
        M, F, T = stft_signals.shape
        _predictions = []
        for T in range(0, T, 10):
            stft_signal = stft_signals[:, :, T:T+10]
            algo.locate_sources(stft_signal)
            _predictions.append(np.rad2deg(algo.azimuth_recon[0]))
        predictions.append(_predictions)
        # algo.locate_sources(stft_signals)
        # predictions.append(np.rad2deg(algo.azimuth_recon[0]))
    predictions = np.array(predictions)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(audio[0])
        for prediction, interval in zip(predictions, intervals):
            line_x = np.arange(interval[0], interval[1]) / fs
            line_y = np.ones_like(line_x) * prediction
            axs[1].plot(line_x, line_y, 'r')
        axs[1].set_xlim(0, audio.shape[-1] / fs)
        plt.savefig('doa.png')
        plt.close()
    return predictions, intervals

def spatial_audio(audio, translation, rotation, mic_array, fs, nfft):
    doas, intervals = pra_doa(audio, mic_array, fs, nfft, plot=False)
    trajectory_sr = 200
    trajectory_intervals = [interval * trajectory_sr / fs for interval in intervals]
    for doa, interval, trajectory_interval in zip(doas, intervals, trajectory_intervals):
        traj_start = int(trajectory_interval[0])
        traj_end = int(trajectory_interval[1])
        print(traj_start, traj_end, translation.shape, rotation.shape)
        trans = translation[traj_start:traj_end]
        rot = rotation[traj_start:traj_end]

        num_observations = range(0, len(translation), len(doa))
        for i, _doa in zip(num_observations, doa):
            t, r = trans[i], rot[i]
            center = t[:2]
            _doa = np.deg2rad(_doa) + r[2]
            mirror_doa = 2 * np.pi - _doa
            # plot the line from center with doa as angle
            end = center + np.array([np.cos(_doa), np.sin(_doa)]) * 0.1
            mirror_end = center + np.array([np.cos(mirror_doa), np.sin(mirror_doa)]) * 0.1
            print(center, _doa)
            plt.plot([center[0], end[0]], [center[1], end[1]], 'r')
            plt.plot([center[0], mirror_end[0]], [center[1], mirror_end[1]], 'g')
            plt.scatter(center[0], center[1], c='b')
            # add to roll
        print(doa, trans.shape, rot.shape)
        plt.savefig('spatial_audio.png')




