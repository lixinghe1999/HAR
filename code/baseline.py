from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration

from EfficientAT.windowed_inference import EATagger
from limu_bert import LIMU_BERT_Inferencer
from sentence_transformers import SentenceTransformer

from utils.qwen_audio import init_qwen, inference_qwen
from utils.imagebind import init_imagebind, inference_imagebind
from utils.onellm import init_onellm, inference_onellm
from OneLLM.data.data_utils import make_audio_features

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch


def visualize(data, tags, action, idx):
    tags = [t['label'] for t in tags if t['probability'] > 0.1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(data['audio'])
    ax[0].set_title(tags)
    ax[1].plot(data['imu'].T)
    ax[1].set_title(action)
    
    plt.suptitle(data['text'])
    plt.savefig('figs/data/{}.png'.format(idx))
    plt.close()
if __name__ == "__main__":
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='small', choices=['small', 'imagebind', 'qwen', 'poe', 'onellm'])
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--window_sec', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='ego4d')
    args = parser.parse_args()

    if args.dataset == 'egoexo':
        dataset = EgoExo_atomic(window_sec=args.window_sec, modal=['audio', 'imu'])
    else:
        dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_audio_imu.json', 
                                  window_sec=args.window_sec, modal=['audio', 'imu'])

    if args.num_samples == -1:
        num_samples = len(dataset)
    else:
        num_samples = args.num_samples
    
    if args.method == 'small':
        audio_model = EATagger(model_name='mn10_as', device='cuda')
        imu_model = LIMU_BERT_Inferencer(ckpt='resources/0.94_finetune.pth', device='cuda')
        for i in tqdm(range(num_samples)):
            random_idx = random.randint(0, len(dataset))
            data_sample = dataset[random_idx]
            tags, features = audio_model.tag_audio_array(data_sample['audio'], sr=dataset.sr_audio)
            action = imu_model.infer(data_sample['imu'], sr=dataset.sr_imu)
            visualize(data_sample, tags, action, i)
    elif args.method == 'imagebind':
        assert args.window_sec == 10
        imagebind = init_imagebind('cuda')
        correct_A, correct_I = 0, 0
        other_texts = ['The user is walking on the street', 'The user is drinking milk', 'The user is playing guitar', 
                       'The user is playing basketball']
        for i in tqdm(range(num_samples)):
            random_idx = random.randint(0, len(dataset))
            data_sample = dataset[random_idx]
            if len(data_sample['audio'].shape) == 2:
                data_sample['audio'] = data_sample['audio'][:1].astype(np.float32)
            else:
                data_sample['audio'] = data_sample['audio'][None, :].astype(np.float32)
            print(data_sample['audio'].shape, data_sample['imu'].shape, data_sample['text'])
            cross_similarity_AT, cross_similarity_IT = inference_imagebind(imagebind, 
                                [data_sample['text']] + other_texts,
                                [(data_sample['audio'], sr_audio)],
                                [data_sample['imu'][:, ::int(sr_imu//200)].T], 'cuda')
            if torch.argmax(cross_similarity_AT[0]) == 0:
                correct_A += 1
            if torch.argmax(cross_similarity_IT[0]) == 0:
                correct_I += 1
        print("Accuracy Audio: ", correct_A/num_samples, "Accuracy IMU: ", correct_I/num_samples)
    elif args.method == 'qwen':
        import soundfile as sf
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import json
        model, tokenizer = init_qwen()
        text_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to('cuda')
        response_list = []
        for i in tqdm(range(num_samples)):
            random_idx = random.randint(0, len(dataset))
            data_sample = dataset[random_idx]
            sf.write('tmp.wav', data_sample['audio'][:1, ::dataset.sr_audio//16000].T, samplerate=16000, )
            response = inference_qwen(model, tokenizer)
            os.remove('tmp.wav')
            gt_text_embedding = text_model.encode([data_sample['text']])[0]
            pred_text_embedding = text_model.encode([response])[0]
            dot_product = np.dot(gt_text_embedding, pred_text_embedding)
            cosine_similarity = dot_product / (np.linalg.norm(gt_text_embedding) * np.linalg.norm(pred_text_embedding))

            response_list.append({'text': data_sample['text'], 'result_audio': response, 'cosine_similarity_audio': float(cosine_similarity)})     
        sum_similarity = 0
        for r in response_list:
            sum_similarity += r['cosine_similarity_audio']
        print("Average Similarity: ", sum_similarity/num_samples)
        json.dump(response_list, open('qwen_response.json', 'w'), indent=4)
    else:
        import json
        import os
        import soundfile as sf

        assert args.window_sec == 2 # OneLLM only supports 2 seconds window (IMU)
        model, target_dtype = init_onellm()
        text_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to('cuda')
        response_list = []
        for i in tqdm(range(num_samples)):
            random_idx = random.randint(0, len(dataset))
            
            # random_idx = i
            data_sample = dataset[random_idx]
            imu = torch.tensor(data_sample['imu'][None, :, ::dataset.sr_imu//200].astype(np.float32))
            sf.write('tmp_onellm.wav', data_sample['audio'][:1, ::dataset.sr_audio//16000].T, samplerate=16000, )
            audio = torch.tensor(make_audio_features('tmp_onellm.wav', mel_bins=128).transpose(0, 1)[None, None])
            result_imu = inference_onellm(model, target_dtype, imu, modal=['imu'])
            result_audio = inference_onellm(model, target_dtype, audio, modal=['audio'])

            gt_text_embedding = text_model.encode([data_sample['text']])[0]
            audio_text_embedding = text_model.encode(result_audio)[0]
            imu_text_embedding = text_model.encode(result_imu)[0]

            dot_product = np.dot(gt_text_embedding, audio_text_embedding)
            cosine_similarity_audio = dot_product / (np.linalg.norm(gt_text_embedding) * np.linalg.norm(audio_text_embedding))
            dot_product = np.dot(gt_text_embedding, imu_text_embedding)
            cosine_similarity_imu = dot_product / (np.linalg.norm(gt_text_embedding) * np.linalg.norm(imu_text_embedding))
            response_list.append({'text': data_sample['text'], 
                                  'result_imu': result_imu[0], 
                                  'result_audio': result_audio[0],
                                  'cosine_similarity_audio': float(cosine_similarity_audio), 
                                  'cosine_similarity_imu': float(cosine_similarity_imu)})
            print(data_sample['text'], result_audio[0], result_imu[0])
        json.dump(response_list, open('onellm_response.json', 'w'), indent=4)