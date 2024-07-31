import numpy as np
from scipy.optimize import minimize
import librosa

def gccphat(audio1, audio2, gcc_phat_len):
    n = audio1.shape[0] + audio2.shape[0] 
    X = np.fft.rfft(audio1, n=n)
    Y = np.fft.rfft(audio2, n=n)
    R = X * np.conj(Y)
    cc = np.fft.irfft(R / (1e-6 + np.abs(R)),  n=(1 * n))
    cc = np.concatenate((cc[-gcc_phat_len:], cc[: gcc_phat_len+1])).astype(np.float32)
    return cc
def calculate_range(audio, mic_center, mic_array, fs):
    C, N = audio.shape
    intervals = librosa.effects.split(y=audio, top_db=30, ref=1)
    n_windows = np.shape(intervals)[0]
    ranges = []
    for i in range(n_windows):
        start = intervals[i][0]
        end = intervals[i][1]
        data = audio[:, start:end]
        TDOA = []
        for i in range(1, C):
            gccphat_result = gccphat(data[0], data[i], gcc_phat_len=int(fs * 0.002))
            argmax = np.argmax(gccphat_result)
            tdoa = (argmax - (int(fs * 0.002))) 
            TDOA.append(tdoa)
        print(TDOA)
        # ranges.append(range_optimize(TDOA, mic_center, mic_array))
    ranges = np.array(ranges)
    return ranges





