'''
convert annotations into the below
[
  {
    "id": "997bb945-628d-4724-b370-b84de974a19f",
    "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
      },
      {
        "from": "gpt",
        "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
      },
    ]
  },
  ...
]
we add another line: time_start, time_end
'''
import os
import json
from tqdm import tqdm
Prompt1 = "<image>\nDescribe the atomic action the user are doing."
Prompt2 =  "<image>\nDescribe the atomic action the user are doing sequentially, separated by ."
Prompt_condition = "Note that the user is performing {}"
Prompt3 = "<image>\nDescribe the last atomic action the user are doing sequentially. Note that the user already did the following atomic actions: \n: {}"
def converter_single(dataset, time_window=4, condition='task_name'):
    '''
    Each conversation only talks about one atomic action
    '''
    templates = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        template = {}
        template['id'] = data['id']
        template['image'] = data['root_dir']
        template['time_start'] = data['timestamp'] - time_window//2
        template['time_end'] = data['timestamp'] + time_window//2
        if condition is not False:
            prompt = Prompt1 + ' ' + Prompt_condition.format(data[condition])
        else: 
            prompt = Prompt1
        template['conversations'] = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': data['text']}]
        templates.append(template)
        # break
    with open('playground/egoexo/prompts/atomic_1.json', 'w') as f:
        json.dump(templates, f, indent=4)

def converter_multiple(dataset, n_action=2, time_window=2):
    '''
    Each conversation talks about multiple (continuously) atomic actions
    '''
    templates = []
    for i in range(0, len(dataset) - n_action + 1, n_action):
        template_image = dataset[i]['root_dir']
        within_same_take = True
        for j in range(n_action):
            if dataset[i+j]['root_dir'] != template_image:
                within_same_take = False
                break
        if not within_same_take:
            continue
        template = {}
        template['id'] = dataset[i]['id']
        template['image'] = dataset[i]['root_dir']
        template['time_start'] = dataset[i]['timestamp'] - time_window//2
        template['time_end'] = dataset[i+n_action-1]['timestamp'] + time_window//2
        template['conversations'] = []
        texts = []
        for j in range(n_action):
            data = dataset[i+j]
            texts.append(data['text'])
        template['conversations'] += [{'from': 'human', 'value':Prompt2}, {'from': 'gpt', 'value': '\n'.join(texts)}]
        
        templates.append(template)
        break
    with open('playground/egoexo/prompts/atomic_{}.json'.format(n_action), 'w') as f:
        json.dump(templates, f, indent=4)

class EgoExo_atomic():
    def __init__(self, data_dir = 'playground/egoexo', split='train'):
        self.data_dir = data_dir
        self.meta = json.load(open(os.path.join(data_dir, 'takes.json')))
        self.takes_by_uid = {x["take_uid"]: x for x in self.meta}
        print('Total number of takes:', len(self.takes_by_uid))

        self.annotation_dir = os.path.join(data_dir, 'annotations')
        self.atomic = os.path.join(self.annotation_dir, 'atomic_descriptions_{}.json'.format(split))
        self.atomic = json.load(open(self.atomic))['annotations']
        self.all_descriptions = []
        for take_uid, xs in self.atomic.items():
            '''make sure the file exist'''
            if take_uid not in self.takes_by_uid:
                continue
            take_meta = self.takes_by_uid[take_uid]
            if take_meta['vrs_relative_path'] == None:
                continue
            # take_path = os.path.join(self.data_dir, take_meta['root_dir'])
            # if not os.path.exists(take_path):
            #     continue
            for x in xs:
                if x['rejected']:
                    continue
                descriptions = x['descriptions']
                for description in descriptions:
                    if description['subject'] != 'C':
                        continue
                    self.all_descriptions.append((take_uid, description))
        print('Total number of atomic descriptions:', len(self.all_descriptions))
    def __len__(self):
        return len(self.all_descriptions)
    def __getitem__(self, idx):
        dict_out = {}
        take_uid, description = self.all_descriptions[idx]
        take_meta = self.takes_by_uid[take_uid]

        text = description['text'].replace('C ', 'The user ')
        timestamp = description['timestamp']

        dict_out['text'] = text
        dict_out['id'] = idx
        dict_out['take_uid'] = take_uid
        dict_out['timestamp'] = timestamp
        dict_out['root_dir'] = take_meta['root_dir']
        dict_out['task_name'] = take_meta['task_name']
        return dict_out

if "__main__" == __name__:
    dataset = EgoExo_atomic(split='train')
    converter_single(dataset)
    # converter_multiple(dataset, n_action=2)
    print("done")
