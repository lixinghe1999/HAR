from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from egoexo.aoa import init_music, inference_music
from egoexo.range import calculate_range
from utils.qwen_vl import init_qwenvl, inference_qwenvl
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps # VAD
from utils.imagebind import init_imagebind, inference_imagebind, imagebind_dataset
import random
import soundfile as sf
import torch
import matplotlib.pyplot as plt
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='prune_silence')
    args = parser.parse_args()
    other_texts = ['The user is walking on the street', 'The user is drinking milk', 'The user is playing guitar', 
                       'The user is playing basketball']
    if args.mode == 'prune_silence':

        # step 1: get the negative samples and keep the sounding ones
        # dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic.json', modal=['audio', 'image'], window_sec=5)
        # dataset.prune_slience('resources/egoexo_atomic_prune.json')
        # dataset.negative()
        # dataset.save_json('resources/egoexo_atomic_negative.json')
        # dataset.prune_slience('resources/egoexo_atomic_negative_prune.json')

        # step 2: get the audio stat
        # dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_narration.json', modal=['audio', 'imu'], window_sec=2)
        # dataset.audio_stat('resources/ego4d_narration_stat.json')
        # step3: use the stat to prune the silence
        dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_narration_stat.json', window_sec=2)
        dataset.audio_prune('resources/ego4d_narration_prune.json', snr_thres=25)
    else:
        # model, tokenizer = init_qwenvl()
        # imagebind = init_imagebind('cuda')
        # dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic_negative_prune.json', modal=['audio', 'image'], window_sec=5)
        # dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic_prune.json', modal=['audio',], window_sec=2)
        dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_narration_prune.json', modal=['audio', 'imu'], window_sec=2)
        dataset = imagebind_dataset(dataset)
        dataset.save_json('resources/ego4d_narration_prune_cosine.json')
        # cosine_diffs = []
        # for i in range(10):
        #     data = dataset[random.randint(0, len(dataset)-1)]
        #     # image = data['image']
        #     # import torchvision.transforms as transforms
        #     # to_pil = transforms.ToPILImage()
        #     # to_pil(image[0]).save('tmp.jpg')
        #     # response = inference_qwenvl(model, tokenizer)

        #     embeddings = inference_imagebind(imagebind, [data['text']], [(data['audio'], dataset.sr_audio)], None, [data['image'][0].numpy()], 'cuda')
        #     # cosine_audio_vision =  torch.nn.functional.cosine_similarity(embeddings['audio'], embeddings['vision'], dim=1)
        #     cosine_audio_text = torch.nn.functional.cosine_similarity(embeddings['audio'], embeddings['text'], dim=1).item() # it is high when the audio is related to action
        #     # cosine_text_vision = torch.nn.functional.cosine_similarity(embeddings['text'], embeddings['vision'], dim=1).item()

        #     c1 = (cosine_text_vision - cosine_audio_text) # on text but not on audio
            
        #     cosine_diffs.append([c1, cosine_audio_text, cosine_text_vision])
        #     cosine_audio = inference_imagebind(imagebind, None, [(data['audio'], dataset.sr_audio)], None, [data['image'][0].numpy()], 'vision', 'cuda')['audio'].item()
        #     cosine_text = inference_imagebind(imagebind, [data['text']], None, None, [data['image'][0].numpy()], 'vision', 'cuda')['text'].item()
        #     cosine_diff = cosine_audio - cosine_text
        #     cosine_diffs.append(cosine_diff)
        # plt.plot(cosine_diffs,)
        # # plt.hist(cosine_diffs, bins=20)
        # plt.savefig('figs/cosine_diff.png')