from egoexo_dataset import EgoExo_atomic
from EfficientAT.windowed_inference import EATagger
from limu_bert import LIMU_BERT_Inferencer
from utils.multi_channel import ssl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
def init_poe():
    tokens = {
        'b': 'DILCAjW1zCzFdnIs8FIQGQ%3D%3D', 
        'lat': 'jOG2v9ajZfE9Dy3FPWdwiR%2B6E0N9Ig1Ndx85GQMdhw%3D%3D'
    }
    from poe_api_wrapper import PoeApi
    client = PoeApi(cookie=tokens)

    bot = 'gpt3_5'
    prompt = "Can you estimate what kind of activity a user is performing based the ambient sound? You will be provided the format like{'start': 10.0, 'end': 11.0, 'tags': [{'idx': the index of sound event, 'label': the name of sound event, 'probability'}]}: . Please respond with a series of potential activities the users performed, the activties should be specific.?"
    return client, bot, prompt
def call_poe(client, bot, prompt, message):
    message = prompt + message
    for chunk in client.send_message(bot, message):
        pass
    print(chunk["text"])
def visualize(data, tags, predict, action, idx):
    tags = [t['label'] for t in tags if t['probability'] > 0.5]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(data['video'][0])
    ax[1].plot(data['audio'][0])
    ax2 = ax[1].twinx()
    ax2.plot(np.arange(0, data['audio'].shape[-1], 48000), predict, c='r')
    ax[1].set_title(tags)
    ax[2].plot(data['imu'])
    ax[2].set_title(action)
    fig.suptitle(','.join([data['parent_task_name'], data['task_name'], data['text']]), fontsize=20)

    plt.savefig('figs/data/{}.png'.format(idx))
    plt.close()
if __name__ == "__main__":
    import random
    audio_model = EATagger(model_name='dymn10_as', device='cuda')
    imu_model = LIMU_BERT_Inferencer(ckpt='0.94_finetune.pth', device='cuda')

    example_imu = np.load('small_dataset/uci/data_20_120.npy')[5, :, :]


    dataset = EgoExo_atomic(window_sec=6, modal=['audio', 'imu', 'video'])
    for i in tqdm(range(1)):
        random_idx = random.randint(0, len(dataset))
        data = dataset[random_idx]
        tags, features = audio_model.tag_audio_array(data['audio'], sr=48000)
        predict, rms = ssl(data['audio'])
        # example_imu = np.load('small_dataset/uci/data_20_120.npy')[45, :, :]
        # example_label = np.load('small_dataset/uci/label_20_120.npy')[45, 0, 0]
        # print(example_label)
        # data['imu'] = example_imu

        action = imu_model.infer(data['imu'], sr=800)
        visualize(data, tags, predict, action, i)
    # client, bot, prompt = init_poe()
    # message = "music"
    # call_poe(client, bot, prompt, message)