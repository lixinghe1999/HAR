from egoexo.egoexo_dataset import EgoExo_atomic, EgoExo
from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Sound, Ego4D_Narration_Sequence, Ego4D_Free
from middleware import imu_middleware, audio_middleware, audio_imu_middleware
import soundfile as sf
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # dataset = Ego4D_Narration(modal=['audio', 'imu',], window_sec=10)
    dataset = Ego4D_Sound(modal=['audio', 'imu',], window_sec=10)
    # dataset = EgoExo_atomic(modal=['audio', 'imu',], window_sec=10)
    print(dataset.scenario_map)
    records = []
    for data in dataset:
        imu = data['imu'] # [acc, gyro]
        audio = data['audio'] 
        text = data['text']
        acc_tag, gyro_tag = imu_middleware.convert(imu[3:], imu[:3])
        audio_tag = audio_middleware.convert(audio)
        audio_imu_tag = audio_imu_middleware.convert(audio, imu)

        record = {}
        record['speech'] = audio_imu_tag
        record['audio'] = audio_tag
        record['imu'] = {'acc': acc_tag, 'gyro': gyro_tag}
        print(text, audio_imu_tag, acc_tag.shape, audio_tag)
        records.append(record)
        break
    # import json
    # with open('ego4d_sound.json', 'w') as f:
    #     json.dump(records, f)

