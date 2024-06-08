'''
this script is used to extract audio from video files
'''
import os
import json
ego4d_meta = '../dataset/ego4d/v2/annotations/ego4d.json'
meta = json.load(open(ego4d_meta))
video_folder = '../dataset/ego4d/v2/audio'
imu_folder = '../dataset/ego4d/v2/imu'
audio_imu_video_count = 0
imu_video_count = 0
audio_video_count = 0
for v in meta['videos']:
    v_meta = v['video_metadata']
    audio_start_sec = v_meta['audio_start_sec']
    audio_duration_sec = v_meta['audio_duration_sec']
    has_imu = v['has_imu']
    has_audio = audio_start_sec is not None
    exist_imu = os.path.exists(os.path.join(imu_folder, v['video_uid'] + '.csv'))
    if has_audio:
        audio_video_count += 1
    if has_imu:
        imu_video_count += 1
    if has_audio and has_imu and exist_imu:
        audio_imu_video_count += 1
        # command = 'python3 -m ego4d.cli.cli --output_directory="~/ecosystem/dataset/ego4d/" --datasets full_scale --video_uids {} -y'.format(v['video_uid'])
        # os.system(command)
        # if not os.path.exists(os.path.join(video_folder, v['video_uid'] + '.mp4')):
        #         print(v['video_uid'], 'not exist')
        # # extract audio from video
        # command = 'ffmpeg -i {}.mp4 -vn -acodec libmp3lame -b:a 192k {}.mp3'.format(os.path.join(video_folder, v['video_uid']), os.path.join(video_folder, v['video_uid']))
        # # make it aac as original
        # # command = 'ffmpeg -i {}.mp4 -vn -acodec copy {}.aac'.format(os.path.join(video_folder, v['video_uid']), os.path.join(video_folder, v['video_uid']))

        # os.system(command)
        # os.system('rm {}.mp4'.format(os.path.join(video_folder, v['video_uid'])))
print('audio_video_count:', audio_video_count)

print('imu_video_count:', imu_video_count)
print('downloaded in folder', len(os.listdir(imu_folder)))
print('audio_imu_video_count:', audio_imu_video_count)
print('downloaded in folder', len(os.listdir(video_folder)))

       