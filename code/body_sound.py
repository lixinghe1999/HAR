from egoexo.egoexo_dataset import EgoExo_atomic
from tqdm import tqdm

def post_process(text):
    print('origial text', text)
    text = text.replace('\n', '-')
    text = text.split('-')
    text = [t.strip() for t in text]
    # if len(text) == 1:
    #     text = [text[0], text[0]]
    print('post process text', text)
    return text
dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic.json', window_sec=4, modal=[])
word_dict = {}
for i in tqdm(range(100)):
    data = dataset[i] 
    sound = post_process(data['sound'])
    for s in sound:
        if s not in word_dict:
            word_dict[s] = 0
        word_dict[s] += 1
print(word_dict)
