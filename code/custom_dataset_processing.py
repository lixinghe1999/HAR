import numpy as np
import librosa
import soundfile as sf
import os
import pandas as pd
from utils.tracking import imu_loading, imu_fusion, imu_tracking, plot_sensor, plot_tracking
sample_rate = 16000
length = 5
# load the audio from the video
def load_audio(video_file, audio_file):
    from moviepy.editor import VideoFileClip
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)
    audio.close()
    video.close()
    return audio
data_file = '../dataset/aiot/Lixing_home-20241106_082431_132'
video_file = data_file + '.mp4'
audio_file = data_file + '.mp3'
imu_file = '../dataset/aiot/Lixing_home-20241106_082431_153.csv'

timestamp, gyroscope, accelerometer, delta_time = imu_loading(imu_file)
timestamp, gyroscope, accelerometer, euler, internal_states, acceleration = imu_fusion(timestamp, gyroscope, accelerometer, delta_time, sample_rate=50)
plot_sensor(timestamp, gyroscope, accelerometer, euler, internal_states)
# audio = load_audio(video_file, audio_file)

# separate the audio into chunks according to the timestamps

# audio = librosa.load(audio_file, sr=sample_rate)[0]
# os.makedirs(data_file, exist_ok=True)
# annotation = data_file + '.txt'
# with open(annotation, 'r') as f:
#     lines = f.readlines()

# activities = '../dataset/aiot/activitiy_attribute.xlsx'
# df = pd.read_excel(activities)
# segment_count = 0
# for line in lines[1:]:
#     line = line.strip().split(',')
#     start, end, har = line[0], line[1], line[2]
#     # find the activity in the activities file
#     attribute = df[df['activity'] == har]
#     # get the column name of the activity (=1)
#     attribute = attribute.columns[attribute.eq(1).any()].to_list()
#     attribute = '_'.join(attribute)

#     start_time = sum(x * int(t) for x, t in zip([3600, 60, 1], start.split(':')))
#     end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(':')))
#     audio_segment = audio[int(start_time*sample_rate):int(end_time*sample_rate)]
#     # save audio_segment into file of 5 seconds
#     for i in range(0, len(audio_segment), sample_rate*length):
#         audio_segment = audio[i:i+sample_rate*length]
#         audio_file_segment = os.path.join(data_file, f'{segment_count},{har},{attribute}.mp3')
#         sf.write(audio_file_segment, audio_segment, sample_rate)
#         segment_count += 1


