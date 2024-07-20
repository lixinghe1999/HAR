'''
We get the interesting sound from general text
'''

import json
import os
import torch
import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.qwen2 import load_model, inference
from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm


if __name__ == '__main__':
    # dataset = EgoExo_atomic()
    # json_dataset = {'text':[], 'task_name':[], 'take_uid':[], 'root_dir':[], 'timestamp':[], 'sound':[]}
    # fname = 'egoexo_atomic_{}.json'

    dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_audio_imu.json', modal=['audio', 'imu'], window_sec=2)
    # dataset.save_json('resources/ego4d_audio_imu.json')
    fname = 'resources/ego4d_audio_imu_qwen_noun.json'

    vad_model = load_silero_vad()

    model, tokenizer = load_model()
    prompt = "As a helpful AI assistant, you will be provided with a sentence describing an action. \
    If the action can produce sound, you will generate the noun that relate to the sound. \
    For example, given the sentence of 'walking on the street', you will generate 'foot step'. \
    If the sentence is 'clapping', you will generate 'hands' without hyphen.\
    If the action described cannot produce any sound, you will simply state 'no sound'."\
    
    # prompt = "As a helpful AI assistant, you will be provided with a sentence describing an action. \
    # Based on the sentence, please tell me the sound that related to the action. \
    # Please said 'no sound' if the action can't produce sound."
    # message=["The subject is running to pass the football by left foot.", "The subject is running to pass the football by right foot."]
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        audio = data['audio']

        # import soundfile as sf
        # print(audio.shape)
        # sf.write('tmp_qwen.wav', audio.T, 16000)
        rms = np.sqrt(np.mean(audio**2))
        max_audio = np.max(np.abs(audio))
        if rms < 0.01 and max_audio < 0.04:
            sound = 'no sound'
        else:
            speech_timestamps = get_speech_timestamps(torch.tensor(audio, dtype=torch.float), vad_model)
            if len(speech_timestamps) != 0:
                sound = 'speech'
            else:
                response = inference(model, tokenizer, prompt, message=[data['text']])
                sound = response
        dataset.add(data['text'], sound, i)
    dataset.save_json(fname)