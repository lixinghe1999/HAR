from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego4d_dataset import Ego4D_Narration
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps # VAD
from utils.imagebind import init_imagebind, inference_imagebind, imagebind_dataset
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
    elif args.mode == 'imagebind':
        # model, tokenizer = init_qwenvl()
        # imagebind = init_imagebind('cuda')
        # dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic_negative_prune.json', modal=['audio', 'image'], window_sec=5)
        # dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic_prune.json', modal=['audio',], window_sec=2)
        dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_narration_prune.json', modal=['audio', 'imu'], window_sec=2)
        dataset = imagebind_dataset(dataset)
        dataset.save_json('resources/ego4d_narration_prune_cosine.json')
        

