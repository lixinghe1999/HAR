import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.qwen2 import load_model, inference
from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm
from EfficientAT.windowed_inference import EATagger


if __name__ == '__main__':
    # dataset = EgoExo_atomic()
    # json_dataset = {'text':[], 'task_name':[], 'take_uid':[], 'root_dir':[], 'timestamp':[], 'sound':[]}
    # fname = 'egoexo_atomic_{}.json'

    dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_rms.json', modal=['audio'])
    audio_model = EATagger(model_name='dymn10_as', device='cuda')

    # dataset.save_json('resources/ego4d.json')
    # dataset.cal_rms()
    dataset.audio_tagging(audio_model)
    keys = ['text', 'video_uid', 'timestamp']
    fname = 'ego4d_narration.json'

    # model, tokenizer = load_model()
    # prompt = "As a helpful AI assistant, you will be provided with a sentence describing an action. If the action can produce sound, you will generate the objects that relate to the sound. For example, given the sentence of 'the user is walking on the street', you will generate 'feet-ground', connecting the two objects by a hyphen. \
    # If there are only one object involved like 'clapping', you will generate 'hands' without hyphen.\
    # If the action described cannot produce any sound, you will simply state 'no sound'.\
    # For each sentence, please list all the possibilities and separate by line break."
    # message=["The subject is running to pass the football by left foot.", "The subject is running to pass the football by right foot."]
    # json_dataset = []
    # for i in tqdm(range(len(dataset))):
    #     data = dataset[i]
        # json_dataset.append({key:data[key] for key in keys})
        # response = inference(model, tokenizer, prompt, message=[data['text']])
        # json_dataset[-1]['sound'] = response
    # json.dump(json_dataset, open(fname, 'w'), indent=4)
        