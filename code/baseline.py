from egoexo.egoexo_dataset import EgoExo_atomic
from EfficientAT.windowed_inference import EATagger
from limu_bert import LIMU_BERT_Inferencer
from egoexo.multi_channel import ssl

from utils.qwen_audio import load_model, inference

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch

from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
def init_imagebind(device):
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model
def inference_imagebind(model, text_list, audio_paths, imu_paths, device):
    inputs = {
        ModalityType.TEXT: imagebind_data.load_and_transform_text(text_list, device),
        ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(audio_paths, device),
        ModalityType.IMU: imagebind_data.load_and_transform_imu_data(imu_paths, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)
    #cross_similarity = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
    # print("Audio x Text: ", cross_similarity)
    cross_similarity_AT = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
    cross_similarity_IT = torch.softmax(embeddings[ModalityType.IMU] @ embeddings[ModalityType.TEXT].T, dim=-1)
    return cross_similarity_AT, cross_similarity_IT


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
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='small', choices=['small', 'imagebind', 'qwen', 'poe'])
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    dataset = EgoExo_atomic(window_sec=10, modal=['audio', 'imu', 'video'])
    if args.num_samples == -1:
        num_samples = len(dataset)
    else:
        num_samples = args.num_samples
    if args.method == 'small':
        audio_model = EATagger(model_name='dymn10_as', device='cuda')
        imu_model = LIMU_BERT_Inferencer(ckpt='resources/0.94_finetune.pth', device='cuda')
        for i in tqdm(range(num_samples)):
            random_idx = random.randint(0, len(dataset))
            data_sample = dataset[random_idx]
            tags, features = audio_model.tag_audio_array(data_sample['audio'], sr=48000)
            predict, rms = ssl(data_sample['audio'])
            action = imu_model.infer(data_sample['imu'], sr=800)
            # visualize(data_sample, tags, predict, action, i)

    elif args.method == 'imagebind':
        imagebind = init_imagebind('cuda')
        correct_A, correct_I = 0, 0
        other_texts = ['The user is walking on the street', 'The user is drinking milk', 'The user is playing guitar', 'The user is playing basketball']
        for i in tqdm(range(num_samples)):
            random_idx = random.randint(0, len(dataset))
            data_sample = dataset[random_idx]
            cross_similarity_AT, cross_similarity_IT = inference_imagebind(imagebind, [data_sample['text']] + other_texts,
                                [(data_sample['audio'][:1].astype(np.float32), 48000)],
                                [data_sample['imu'][::4]], 'cuda')
            if torch.argmax(cross_similarity_AT[0]) == 0:
                correct_A += 1
            if torch.argmax(cross_similarity_IT[0]) == 0:
                correct_I += 1
        print("Accuracy Audio: ", correct_A/num_samples, "Accuracy IMU: ", correct_I/num_samples)
    else:
        import soundfile as sf
        import os
        model, tokenizer = load_model()
        response_list = []
        for i in tqdm(range(num_samples)):
            random_idx = random.randint(0, len(dataset))
            data_sample = dataset[random_idx]
            sf.write('tmp.flac', data_sample['audio'][:1, ::3].T, samplerate=16000, )
            response = inference(model, tokenizer)
            os.remove('tmp.flac')
            response_list.append([response, data_sample['text']])
        print(response_list)