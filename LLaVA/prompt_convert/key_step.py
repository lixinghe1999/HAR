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
def converter(dataset, time_window=2, fname = 'playground/egoexo/prompts/keystep.json'):
    templates = []
    for i in range(len(dataset)):
        data = dataset[i]
        template = {}
        template['id'] = data['id']
        template['image'] = data['root_dir']
        template['time_start'] = data['start_time']
        template['time_end'] = data['end_time']
        template['conversations'] = [{'from': 'human', 'value': "<image>\nDescribe the atomic action the user are doing."}, {'from': 'gpt', 'value': data['text']}]
        print(template)
        templates.append(template)
        break
    with open(fname, 'w') as f:
        json.dump(templates, f, indent=4)

class EgoExo_keystep():
    def __init__(self, data_dir = 'playground/egoexo', split='train'):
        self.data_dir = data_dir
        self.meta = json.load(open(os.path.join(data_dir, 'takes.json')))
        self.takes_by_uid = {x["take_uid"]: x for x in self.meta}
        print('Total number of takes:', len(self.takes_by_uid))

        self.annotation_dir = os.path.join(data_dir, 'annotations')
        self.keystep = os.path.join(self.annotation_dir, 'keystep_{}.json'.format(split))
        self.keystep = json.load(open(self.keystep))

        self.all_descriptions = []
        for take_uid, xs in self.keystep['annotations'].items():
            scenario = xs['scenario']
            # print('Scenario:', scenario)
            for segment in xs['segments']:
                segment['scenario'] = scenario
                self.all_descriptions.append((take_uid, segment))
        print('Total number of keystep descriptions:', len(self.all_descriptions))
    def __len__(self):
        return len(self.all_descriptions)
    def __getitem__(self, idx):
        dict_out = {}
        take_uid, description = self.all_descriptions[idx]
        take_meta = self.takes_by_uid[take_uid]

        dict_out['take_uid'] = take_uid
        dict_out['scenario'] = description['scenario']
        dict_out['start_time'] = description['start_time']
        dict_out['end_time'] = description['end_time']
        dict_out['text'] = description['step_description']
        dict_out['step_id'] = description['step_id']
        dict_out['root_dir'] = take_meta['root_dir']
        dict_out['id'] = idx

        return dict_out

if __name__ == '__main__':
    dataset = EgoExo_keystep()
    converter(dataset)
    print('Done')