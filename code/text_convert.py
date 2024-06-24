import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from code.utils.qwen2 import load_model, inference
from tqdm import tqdm

class Ego4D():
    def __init__(self, path='../dataset/ego4d/v2/',):
        self.data = json.load(open(path, 'r', encoding="utf-8"))        
    def __len__(self):
        return len(self.data)       
    def __getitem__(self, idx):
        data = self.data[idx]
        data['timestamp'] = (data['window_end'] + data['window_start'])//2
        return data
class Ego4D_Moment():
    def __init__(self, folder='../dataset/ego4d/v2/', split='train'):
        self.folder = folder
        moments = json.load(open(self.folder + 'annotations/moments_{}.json'.format(split), 'r', encoding="utf-8"))['videos']
        self.moments_list = []
        avg_len = []
        for v_data in moments:
            moment_list = []
            for moment_data in v_data['clips']:
                for annotation in moment_data['annotations']:
                    labels = annotation['labels']
                    for label in labels:
                        moment_list.append(label)
            if len(moment_list) > 0:
                self.moments_list += [
                    (   v_data['video_uid'],
                        m_t['label'],
                        m_t["video_start_time"],
                        m_t["video_end_time"],
                    )
                    for m_t in moment_list
                ]
                avg_len.append(len(self.moments_list))
        print(f"Number of moment {len(self.moments_list)}")
        print(f"Avg. moment number for each video {sum(avg_len)/len(avg_len)}")
    def __len__(self):
        return len(self.moments_list)       
    def __getitem__(self, idx):
        dict_out = {}
        video_uid, text, start_time, end_time = self.moments_list[idx]
        dict_out['video_uid'] = video_uid
        dict_out['text'] = text
        dict_out['start_time'] = start_time
        dict_out['end_time'] = end_time
        return dict_out
    
class EgoExo_atomic():
    def __init__(self, data_dir = '../dataset/egoexo', split='train', window=4):
        self.data_dir = data_dir
        self.window = window
        self.meta = json.load(open(os.path.join(data_dir, 'takes.json')))
        self.takes_by_uid = {x["take_uid"]: x for x in self.meta}
        print('Total number of takes:', len(self.takes_by_uid))

        self.annotation_dir = os.path.join(data_dir, 'annotations')
        self.atomic = os.path.join(self.annotation_dir, 'atomic_descriptions_{}.json'.format(split))
        self.atomic = json.load(open(self.atomic))['annotations']
        self.all_descriptions = []
        self.filter_atomic()
        for take_uid, xs in self.atomic.items():
            for x in xs:
                if x['rejected']:
                    continue
                descriptions = x['descriptions']
                for description in descriptions:
                    if description['subject'] != 'C':
                        continue
                    self.all_descriptions.append((take_uid, description))
        print('Total number of atomic descriptions:', len(self.all_descriptions))
    def filter_atomic(self):
        new_atomic = {}
        for take_uid, xs in self.atomic.items():
            '''make sure the file exist'''
            if take_uid not in self.takes_by_uid:
                continue
            take_meta = self.takes_by_uid[take_uid]
            if take_meta['vrs_relative_path'] == None:
                continue
            new_atomic[take_uid] = xs
        self.atomic = new_atomic
    def __len__(self):
        return len(self.all_descriptions)
    def __getitem__(self, idx):
        dict_out = {}
        take_uid, description = self.all_descriptions[idx]
        take_meta = self.takes_by_uid[take_uid]

        text = description['text'].replace('C ', 'The user ')
        timestamp = description['timestamp']
        dict_out['timestamp'] = timestamp
        dict_out['text'] = text
        dict_out['id'] = idx
        dict_out['take_uid'] = take_uid
        dict_out['time_start'] = timestamp - self.window//2
        dict_out['time_end'] = timestamp + self.window//2

        dict_out['root_dir'] = take_meta['root_dir']
        dict_out['task_name'] = take_meta['task_name']
        return dict_out

if __name__ == '__main__':
    # dataset = EgoExo_atomic()
    # json_dataset = {'text':[], 'task_name':[], 'take_uid':[], 'root_dir':[], 'timestamp':[], 'sound':[]}
    # fname = 'egoexo_atomic_{}.json'

    dataset = Ego4D('../dataset/ego4d/ego4d_audio.json')
    json_dataset = {'text':[], 'video_uid':[], 'timestamp':[], 'sound':[]}
    fname = 'ego4d_narration_{}.json'

    # dataset = Ego4D_Moment()
    # json_dataset = {'text':[], 'video_uid':[], 'start_time':[], 'end_time':[], 'sound':[]}
    # fname = 'ego4d_moment_{}.json'

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
    for i in tqdm(range(len(dataset))):
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
        