from egoexo.egoexo_dataset import EgoExo_atomic, EgoExo
from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Sound, Ego4D_Narration_Sequence, Ego4D_Free
from middleware import imu_middleware, audio_middleware, audio_imu_middleware
import soundfile as sf
import pandas as pd

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # dataset = Ego4D_Narration(modal=['audio', 'imu',], window_sec=10)
    capture24 = pd.read_csv('./resources/capture24_label_count.csv')
    print(capture24)
    # dataset = Ego4D_Sound(modal=['audio', 'imu',], window_sec=10)
    # dataset = EgoExo_atomic(modal=['audio', 'imu',], window_sec=10)
    # print(dataset.scenario_map)
    # for data in dataset:
    #     imu = data['imu'] # [acc, gyro]
    #     audio = data['audio'] 
    #     text = data['text']
    #     acc_tag, gyro_tag = imu_middleware.convert(imu[3:], imu[:3])
    #     audio_tag = audio_middleware.convert(audio)
    #     audio_imu_tag = audio_imu_middleware.speech_detection(audio, imu)

    #     print(text, audio_imu_tag, acc_tag.shape, audio_tag)
    #     break


