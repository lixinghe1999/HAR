from silero_vad import load_silero_vad, read_audio, get_speech_timestamps # VAD
import numpy as np
class Speech_filter():
    def __init__(self):
        self.silero_vad = load_silero_vad()
    def convert(self, audio, imu):
        '''
        audio: [N]
        imu: [6, N]
        '''
        speech_timestamp = get_speech_timestamps(audio, self.silero_vad)
        if len(speech_timestamp) > 0: # detected speech
            max_cosine_similarity = 0
            for timestamp in speech_timestamp:
                start_stamp = timestamp['start']
                end_stamp = timestamp['end']
                audio_segment = audio[start_stamp:end_stamp]
                down_sample_rate = int(16000 / 200)

                imu_stamp_start = int(start_stamp / down_sample_rate)
                imu_stamp_end = int(end_stamp / down_sample_rate)
                imu = imu[:3, imu_stamp_start:imu_stamp_end] # accel only
                imu_norm = np.linalg.norm(imu, axis=0) 
                down_sample_audio = audio_segment[::down_sample_rate]

                cosine_similarity = np.dot(imu_norm, down_sample_audio) / (np.linalg.norm(imu_norm) * np.linalg.norm(down_sample_audio))
                print('cosine_similarity:', cosine_similarity)
                if cosine_similarity > max_cosine_similarity:
                    max_cosine_similarity = cosine_similarity
            if max_cosine_similarity > 0.5:
                return 'speech'
            else:
                return 'ambient speech'
        else:
            return 'no speech'
        