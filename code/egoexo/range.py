from .multi_channel import fs, mic_center, mic_array, gccphat
import numpy as np
from scipy.optimize import minimize
import librosa
def calculate_range(audio):
    C, N = audio.shape
    intervals = librosa.effects.split(y=audio, top_db=20, ref=1)
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
            tdoa = (argmax - (int(fs * 0.002))) / fs
            TDOA.append(tdoa)
        ranges.append(range_optimize(TDOA))
    ranges = np.array(ranges)
    return ranges

def range_optimize(TDOA):
    def compute_tdoa_diff(source_pos, mic_positions, tdoa, speed_of_sound):
        d0 = np.linalg.norm(source_pos - mic_positions[0])
        distances = np.linalg.norm(source_pos - mic_positions[1:], axis=1)
        calculated_tdoa = (distances - d0) / speed_of_sound
        return np.sum((calculated_tdoa - tdoa)**2)
    initial_guess = mic_center
    result = minimize(compute_tdoa_diff, initial_guess, args=(mic_array.T, TDOA, 343))
    return result.x

def range_lls(TDOA):
    dd = TDOA * 343
    n_mics = mic_array.shape[0]

    # Reference microphone (Mic 1)
    ref_mic = mic_array[0]

    # Construct matrix A and vector b for the LLS problem
    A = np.zeros((n_mics-1, 3))
    b = np.zeros(n_mics-1)

    for i in range(1, n_mics):
        A[i-1, :] = 2 * (mic_array[i, :] - ref_mic)
        b[i-1] = (dd[i-1]**2 - np.sum(mic_array[i, :]**2) + np.sum(ref_mic**2))

    # Solve the LLS problem using np.linalg.lstsq
    source_position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return source_position



