from ego4d.ego4d_dataset import Ego4D_Narration
import soundfile as sf
import numpy as np
import os
import torch
import librosa
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='embeddings')
    args = parser.parse_args()
    '''
    Mode == embeddings: extract the text embedding for further processing
    Mode == scenario_similarity: inter and intra similarity of the scenario
    Mode == select_activity: select the top 50 samples for each activity
    Mode == scenario_visualization: visualize the scenario by the embeddings
    Mode == ambient: add noise to the audio embeddings and evaluate the similarity
    '''
    if args.mode == 'embeddings':
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')

        # capture24 = pd.read_csv('./resources/capture24_label_count.csv')
        # clean_annotation = []
        # for annotation, count in zip(capture24['annotation'], capture24['count']):
        #     annotations = annotation.split(';')
        #     annotation = re.sub(r'\d+', '', annotations[-2])
        #     clean_annotation.append(annotation)

        # embeddings = model.encode(clean_annotation)
        # np.save('./resources/capture24_label_embedding.npy', embeddings)

        # from sklearn.cluster import KMeans
        # from sklearn.metrics import pairwise_distances_argmin_min
        # kmeans = KMeans(n_clusters=30, random_state=42)
        # kmeans.fit(embeddings)
        # cluster_centers = kmeans.cluster_centers_
        # # Step 3: Find the nearest data points to each cluster center
        # closest_indices, _ = pairwise_distances_argmin_min(cluster_centers, embeddings)
        # print(closest_indices)
        # print([clean_annotation[i] for i in closest_indices])

        # # plot the embeddings by TSNE and use color from kmeans
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, random_state=42)
        # embeddings_2d = tsne.fit_transform(embeddings)
        # import matplotlib.pyplot as plt
        # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans.labels_)
        # plt.savefig('./figs/capture24_label_embedding.png')

        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()

        text_embeddings = []; audio_embeddings = []
        dataset = Ego4D_Narration(modal=['audio', 'imu'], window_sec=10)
        # dataset_loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=8)
        for i in tqdm(range(len(dataset))):
        # for i, data_sample in tqdm( enumerate(dataset_loader)):
            data_sample = dataset.__getitem__(i)
            audio = data_sample['audio']
            text = data_sample['text']
            # audio = torch.nn.functional.upsample(audio.unsqueeze(1), scale_factor=3, mode='linear', align_corners=True).squeeze()
            audio = librosa.resample(y=audio, orig_sr=16000, target_sr=48000)[None, :]

            audio_embedding = model.get_audio_embedding_from_data(x=audio, use_tensor=False)
            audio_embeddings.append(audio_embedding)

            text_embedding = model.get_text_embedding(data_sample['text'])
            text_embeddings.append(text_embedding)

        audio_embeddings = np.concatenate(audio_embeddings, axis=0)   
        np.save('./resources/ego4d_audio_embedding.npy', audio_embeddings)

        text_embeddings = np.concatenate(text_embeddings, axis=0)   
        np.save('./resources/ego4d_narration_embedding.npy', text_embeddings)
        print(audio_embeddings.shape, text_embeddings.shape)
    
    elif args.mode == 'scenario_similarity':
        dataset = Ego4D_Narration(modal=['audio', 'imu'], window_sec=10)
        text_embeddings = np.load('./resources/ego4d_narration_embedding.npy')

        scenario_embeddings = {}
        for i in range(len(dataset)):
            data_sample = dataset.window_idx[i]
            scenario = data_sample['scenario']
            for s in scenario:
                if s not in scenario_embeddings:
                    scenario_embeddings[s] = []
                scenario_embeddings[s].append(i)
        scenario_names = list(scenario_embeddings.keys())
        scenario_similarities = []
        for s in scenario_names:
            scenario = dataset.scenario_map[s].replace('/', ', ')
            idxs = scenario_embeddings[s]
            idxs = np.random.choice(idxs, 500)
            text_embeddings_scenario = text_embeddings[idxs]
            all_idxs = np.arange(len(dataset))
            random_idxs = np.random.choice(all_idxs, len(idxs), replace=False)
            random_embeddings = text_embeddings[random_idxs]

            intra_cosine_similarity = np.dot(text_embeddings_scenario, text_embeddings_scenario.T)
            intra_cosine_similarity = np.mean(intra_cosine_similarity)

            inter_cosine_similarity = np.dot(text_embeddings_scenario, random_embeddings.T)
            inter_cosine_similarity = np.mean(inter_cosine_similarity)
            print(s, scenario, intra_cosine_similarity, inter_cosine_similarity)
            scenario_similarities.append([intra_cosine_similarity, inter_cosine_similarity])
        np.savetxt('./resources/ego4d_scenario_similarity.txt', scenario_similarities, fmt='%f')

    elif args.mode == 'select_activity':

        dataset = Ego4D_Narration(modal=['audio', 'imu'], window_sec=10)
        text_embeddings = np.load('./resources/ego4d_narration_embedding.npy')

        import laion_clap   
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()
        import json
        activity_motion_attributes = json.load(open('./resources/ego4d_activity_basic.json'))
        activity_candidates = list(activity_motion_attributes.keys())
        
        activity_embeddings = model.get_text_embedding(activity_candidates)
        cosine_similarity = np.dot(text_embeddings, activity_embeddings.T)
        # for each activity, find the top 50 samples
        top_k = 50
        dataset_folder = '../dataset/ego4d/sampled/'
        for i in range(len(activity_candidates)):
            argmax_activity = activity_candidates[i]
            idxs = np.argsort(cosine_similarity[:, i])[-top_k:]
            average_similarity = np.mean(cosine_similarity[idxs, i])
            print(average_similarity)
            # only keep the idxs with the > 0.7
            idxs = idxs[np.where(cosine_similarity[idxs, i] > 0.7)]

            print(argmax_activity, len(idxs))
            dataset_folder_activity = f'../dataset/ego4d/mini/{argmax_activity}'
            os.makedirs(dataset_folder_activity, exist_ok=True)
            for j, idx in enumerate(idxs):
                imu_activity = ', '.join(activity_motion_attributes[argmax_activity])
                data_sample = dataset.window_idx[idx]
                scenario = data_sample['scenario']
                if len(scenario) > 3: # meaningless scenario
                    continue
                scenario = ', '.join([dataset.scenario_map[s].replace('/', ' or ') for s in scenario])
                text = data_sample['text']
                
                audio_name = f'{dataset_folder_activity}/{j}_{scenario}_{argmax_activity}_{text}_{imu_activity}.wav'
                imu_name = audio_name.replace('.wav', '.npy')

                data_sample = dataset.__getitem__(idx)
                audio = data_sample['audio']
                imu = data_sample['imu']
                np.save(imu_name, imu[:])
                sf.write(audio_name, audio, 16000)

    elif args.mode == 'scenario_visualization':
        dataset = Ego4D_Narration(modal=['audio', 'imu'], window_sec=10)
        scenario_embeddings = {}
        for i in range(len(dataset)):
            data_sample = dataset.window_idx[i]
            scenario = data_sample['scenario']
            for s in scenario:
                if s not in scenario_embeddings:
                    scenario_embeddings[s] = []
                scenario_embeddings[s].append(i)
        scenario_names = list(scenario_embeddings.keys())
        # sort the scenario_name by the number of samples
        scenario_names = sorted(scenario_names, key=lambda x: len(scenario_embeddings[x]), reverse=True)
        # scenario_names = scenario_names[-5:]
        scenario_names = np.random.choice(scenario_names, 50)

        all_embeddings = {'audio': [], 'imu': [], 'text': []}; all_labels = []
        from imagebind import data
        import torch
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Instantiate model
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(device)
        for s in scenario_names:
            idxs = scenario_embeddings[s]
            idxs = np.random.choice(idxs, 50)
            for idx in idxs:
                data_sample = dataset.__getitem__(idx)
                imu = torch.from_numpy(data_sample['imu'])[None, :]
                audio = torch.from_numpy(data_sample['audio'][None, :])
                text = data_sample['text']
                inputs = {
                    ModalityType.IMU: imu.float().to(device),
                    ModalityType.AUDIO: data.load_and_transform_audio_data([[audio, 16000]], (device)),
                    ModalityType.TEXT: data.load_and_transform_text([text], device),
                }
                with torch.no_grad():
                    embeddings = model(inputs)
                imu_embedding = embeddings[ModalityType.IMU].cpu().numpy()
                audio_embedding = embeddings[ModalityType.AUDIO].cpu().numpy()
                text_embedding = embeddings[ModalityType.TEXT].cpu().numpy()
                all_embeddings['audio'].append(audio_embedding)
                all_embeddings['imu'].append(imu_embedding)
                all_embeddings['text'].append(text_embedding)
                all_labels.append([s])
        all_labels = np.concatenate(all_labels, axis=0)
        for modality in all_embeddings:
            all_embeddings[modality] = np.concatenate(all_embeddings[modality], axis=0)


        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tsne = TSNE(n_components=2, random_state=42)
        for modality in all_embeddings:
            embeddings_2d = tsne.fit_transform(all_embeddings[modality])
            fig, axs = plt.subplots(1, 1, figsize=(6, 4))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels)
            plt.yticks([])
            plt.xticks([])
            plt.tight_layout()
            plt.savefig(f'./figs/ego4d_{modality}_scenario_mapping.png')

        embeddings_2d = tsne.fit_transform(np.concatenate([all_embeddings['audio'], all_embeddings['imu']], axis=1))
        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels)
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(f'./figs/ego4d_multimodal_scenario_mapping.png')

    elif args.mode == 'summary':
        from audio_tag import audio_tagging_inference
        args = argparse.Namespace()
        args.model_name = 'mn10_as'
        args.strides = [2, 2, 2, 2]
        args.head_type = 'mlp'
        args.cuda = True
        args.audio_path = './resources/ego4d_narration_audio.wav'
        args.sample_rate = 32000
        args.window_size = 800
        args.hop_size = 320
        args.n_mels = 128
        args.ensemble = []

        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()
        candidate_motion = ['walking', 'standing', 'lying down']
        candidate_embeddings = model.get_text_embedding(candidate_motion)
        print('extract candidate motion embedding done')

        dataset = Ego4D_Narration(modal=['audio', 'imu'], window_sec=10)
        text_embeddings = np.load('./resources/ego4d_narration_embedding.npy')
        import json
        output_datas = []

        audio_inferencer = audio_tagging_inference(args)
        for i in tqdm(range(len(dataset))):
            data_sample = dataset.__getitem__(i)
            scenario = dataset.window_idx[i]['scenario']
            if len(scenario) > 3: # meaningless scenario
                continue
            scenario = ', '.join([dataset.scenario_map[s] for s in scenario])
            text = data_sample['text']; uid = data_sample['video_uid']

            audio_name = './tmp.wav'
            sf.write(audio_name, data_sample['audio'], 16000)

            preds = audio_inferencer(audio_name)
            preds = {k: float(v) for k, v in preds.items() if v >= 0.2}

            text_embedding = text_embeddings[i]
            argmax_motion = np.argmax(np.dot(text_embedding, candidate_embeddings.T))
            motion_text = candidate_motion[argmax_motion]
            output_data = {
                "scenario": scenario,
                "text": text,
                "audio": preds,
                "motion": motion_text,
                "video_uid": uid
            }
            output_datas.append(output_data)
        os.remove('./tmp.wav')
        with open('./resources/summary.json', 'w') as f:
            json.dump(output_datas, f, indent=4, ensure_ascii=False)