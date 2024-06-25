from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm
import matplotlib.pyplot as plt
def post_process(text):
    print('origial text', text)
    text = text.replace('\n', '-')
    text = text.split('-')
    text = [t.strip() for t in text]
    # if len(text) == 1:
    #     text = [text[0], text[0]]
    print('post process text', text)
    return text
# dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic.json', window_sec=4, modal=[])
# word_dict = {}
# for i in tqdm(range(100)):
#     data = dataset[i] 
#     sound = post_process(data['sound'])
#     for s in sound:
#         if s not in word_dict:
#             word_dict[s] = 0
#         word_dict[s] += 1
# print(word_dict)

dataset = Ego4D_Narration(pre_compute_json='resources/ego4d.json', window_sec=4, modal=['audio', 'imu'])
for data in dataset:
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].plot(data['audio'])
    axs[1].plot(data['imu'].T)
    fig.suptitle(data['text'])

    plt.savefig('tmp.png'.format(data['text']))
    break