import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.qwen2 import load_model, inference
from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm

if __name__ == '__main__':
    # dataset = EgoExo_atomic()
    # json_dataset = {'text':[], 'task_name':[], 'take_uid':[], 'root_dir':[], 'timestamp':[], 'sound':[]}
    # fname = 'egoexo_atomic_{}.json'

    dataset = Ego4D_Narration(modal=['audio'])
    json_dataset = {'text':[], 'video_uid':[], 'timestamp':[], 'sound':[]}
    fname = 'ego4d_narration_{}.json'


    model, tokenizer = load_model()
    prompt = "As a helpful AI assistant, you will be provided with multiple senstences describing an action. If the action can produce sound, \
    you will generate the objects that would create that sound. If there are two objects, like 'walking', you will generate 'feet-ground'. \
    If there are only one object like 'clapping', you will generate 'hands'.\
    If there are additional details provided about the action, you will include those as well like 'sports shoes-gym floor'.\
    If the action described cannot produce any sound, you will simply state 'no sound'.\
    For each sentence, please list all the possibilities and separate by line break."
    message=["The subject is running to pass the football by left foot.", "The subject is running to pass the football by right foot."]
    batch_size = 1
    text_batch = []
    for i in tqdm(range(1)):
        data = dataset[i]
        text_batch.append(data['text'])
        for key in list(json_dataset.keys())[:-1]:
            json_dataset[key].append(data[key])
        if len(text_batch) == batch_size:
            response = inference(model, tokenizer, prompt, message=text_batch)
            # response = response.split('\n')
            # print(text_batch, response)
            # if len(response) == batch_size:
            json_dataset['sound'].append(response)
            text_batch = []
    if len(text_batch) > 0: # remaining
        response = inference(model, tokenizer, prompt, message=text_batch)
        json_dataset['sound'].append(response)
    # assert len(json_dataset['text']) == len(json_dataset['sound'])
    # print(json_dataset)
    json.dump(json_dataset, open(fname.format(batch_size), 'w'), indent=4)
        