from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from egoexo.multi_channel import init_music, inference_music
from EfficientAT.windowed_inference import EATagger
import torch
import soundfile as sf
import librosa
from tqdm import tqdm
import random
import numpy
window_sec = 2
dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_audio_imu.json', window_sec=2, modal=['audio', 'imu'])
filter_type = 'cluster'
def taxonomy2map(taxonomy):
    taxonomy_map = {}
    for i, row in taxonomy.iterrows():
        for word in eval(row['group']):
            taxonomy_map[word] = i
            taxonomy_map[word.lower() + 's'] = i 
    print('number of taxonomy:', len(taxonomy))
    print('number of words in taxonomy:', len(taxonomy_map))
    return taxonomy_map
if filter_type == 'vad':
    model = load_silero_vad()
elif filter_type == 'cluster':
    import pandas as pd
    noun_taxonomy =  '../dataset/ego4d/v2/annotations/narration_noun_taxonomy.csv'
    verb_taxonomy =  '../dataset/ego4d/v2/annotations/narration_verb_taxonomy.csv'
    # load csv file
    noun_taxonomy = pd.read_csv(noun_taxonomy)
    verb_taxonomy = pd.read_csv(verb_taxonomy)

    noun_taxonomy = taxonomy2map(noun_taxonomy)
    verb_taxonomy = taxonomy2map(verb_taxonomy)
elif filter_type == 'music':
    music = init_music()

elif filter_type == 'audiotag':
    audio_model = EATagger(model_name='mn10_as', device='cuda')

num_samples = len(dataset)
for i in tqdm(range(10)):
    idx = i
    data = dataset[idx]

    # VAD filtering
    if filter_type == 'vad':
        audio = librosa.resample(y=data['audio'][0], orig_sr=dataset.sr_audio, target_sr=16000)     
        # sf.write('tmp.wav', audio, 16000)
        # wav = read_audio('tmp.wav') # backend (sox, soundfile, or ffmpeg) required!
        speech_timestamps = get_speech_timestamps(torch.tensor(audio, dtype=torch.float), model)
        if len(speech_timestamps) == 0:
            dataset.add(idx, 'vad', False)
        else:
            dataset.add(idx, 'vad', True)
    if filter_type == 'cluster':
        text = data['text']
        scenario = data['scenario']
        noun_text = [noun_taxonomy[word] for word in text.lower().split() if word in noun_taxonomy]
        verb_text = [verb_taxonomy[word] for word in text.lower().split() if word in verb_taxonomy]
        print(scenario.argmax(), text, noun_text, verb_text)
    if filter_type == 'multi_channel':
        audio = librosa.resample(y=data['audio'], orig_sr=dataset.sr_audio, target_sr=16000)
        music_result = inference_music(music, audio)
        print(music_result)

    if filter_type == 'audiotag':
        audio = librosa.resample(y=data['audio'][0], orig_sr=dataset.sr_audio, target_sr=16000)
        tags, features = audio_model.tag_audio_array(audio, sr=16000)
        print(tags, data['text'])
dataset.save_json('resources/egoexo_atomic_{}.json'.format(filter_type))