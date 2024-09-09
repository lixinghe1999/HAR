from silero_vad import load_silero_vad, get_speech_timestamps # VAD
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.signal as signal

class Multimodal_Processor():
    def __init__(self):
        self.silero_vad = load_silero_vad()

    def resample_imu(self, imu, time_idx=0, time_unit=1e-9, sr_imu=400):
        '''
        audio: [N]
        imu: [N, 7]
        '''
        time_imu = imu[:, time_idx] * time_unit
        sr = 1 / np.diff(time_imu).mean()
        print('sr_imu:', sr)
        # resample imu
        imu = imu[:, 1:].T
        time_imu_new = np.linspace(time_imu[0], time_imu[-1], imu.shape[1])
        imu = np.array([np.interp(time_imu_new, time_imu, imu[i]) for i in range(imu.shape[0])])
        # imu = librosa.resample(y=imu, orig_sr=sr, target_sr=sr_imu)
        print('resampled imu:', imu.shape)
        return imu

    def correlation(self, audio, imu, sr_audio=16000, sr_imu=200, plot=False):
        window_length = 0.1
        hop_length = 0.05
        window_audio = int(window_length * sr_audio)
        hop_audio = int(hop_length * sr_audio)
        window_imu = int(window_length * sr_imu)
        hop_imu = int(hop_length * sr_imu)

        audio_stft = librosa.stft(audio, n_fft=window_audio, hop_length=hop_audio)
        audio_stft = np.abs(audio_stft)

        b, a = signal.butter(3, 10, 'high', fs=sr_imu)
        imu = signal.filtfilt(b, a, imu, axis=1)
        imu = np.linalg.norm(imu, axis=0)
        imu_stft = librosa.stft(imu, n_fft=window_imu, hop_length=hop_imu)
        
        audio = np.abs(audio)
        cosine_similarity = np.dot(audio, imu) / (np.linalg.norm(audio) * np.linalg.norm(imu))
        # print('audio_stft:', audio_stft.shape, 'imu_stft:', imu_stft.shape)
        if plot:
            fig, ax = plt.subplots(4, 1)
            ax[0].imshow(np.abs(audio_stft), aspect='auto')
            ax[1].imshow(np.abs(imu_stft), aspect='auto')
            ax[2].plot(audio)
            ax[3].plot(imu)
            plt.savefig('correlation.png')
        return cosine_similarity

    def speech_detection(self, audio, imu, sr_audio=16000, sr_imu=200):
        '''
        please use silero_vad to detect it in audio
        https://github.com/snakers4/silero-vad
        speech_timestamp = get_speech_timestamps(audio, self.silero_vad)
        '''
        speech_timestamp = get_speech_timestamps(audio, self.silero_vad)
        if len(speech_timestamp) == 0:
            return 'no speech detected'
        else:
            max_correlation = 0
            for segment in speech_timestamp:
                # segment the audio and imu and keep the length same
                start_audio, end_audio = segment
                audio_segment = audio[start_audio:end_audio:int(sr_audio/sr_imu)]
                start_imu = int(start_audio * sr_imu / sr_audio)
                end_imu = start_imu + len(audio_segment)
                imu_segment = imu[:, start_imu:end_imu]
                correlation = self.correlation(audio_segment, imu_segment, sr_audio=sr_imu, sr_imu=sr_imu)
                if correlation > max_correlation:
                    max_correlation = correlation
            if max_correlation > 0.5:
                return 'self speech'
            else:
                return 'ambient speech'
        
        
        
if __name__ == '__main__':
    processor = Multimodal_Processor()
    imu_file = 'tiny_dataset/imu.csv'
    audio_file = 'tiny_dataset/audio.mp3'

    imu = np.loadtxt(imu_file, delimiter=',')
    audio, sr = librosa.core.load(audio_file, sr=16000)
    print('imu:', imu.shape, 'audio:', audio.shape, sr)

    imu = processor.resample_imu(imu, time_idx=0, time_unit=1e-9, sr_imu=400)
    processor.plot_correlation(audio, imu, sr_audio=16000, sr_imu=400)


    
     