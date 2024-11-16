from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Sound, Ego4D_Narration_Sequence, Ego4D_Free
import soundfile as sf
import pandas as pd
import re
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='embeddings')
    args = parser.parse_args()

    if args.mode == 'embeddings':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # text1 = "Places the phone on the table."
        # text2 = "sitting/lying and watching television with TV on as the primary activity" 
        # # Compute embeddings
        # embeddings1 = model.encode(text1)
        # embeddings2 = model.encode(text2)
        # cosine_similarity = np.dot(embeddings1, embeddings2.T)
        # print(cosine_similarity)

        capture24 = pd.read_csv('./resources/capture24_label_count.csv')
        clean_annotation = []
        for annotation, count in zip(capture24['annotation'], capture24['count']):
            annotations = annotation.split(';')
            annotation = re.sub(r'\d+', '', annotations[-2])
            clean_annotation.append(annotation)

        embeddings = model.encode(clean_annotation)
        np.save('./resources/capture24_label_embedding.npy', embeddings)

        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
        kmeans = KMeans(n_clusters=30, random_state=42)
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        # Step 3: Find the nearest data points to each cluster center
        closest_indices, _ = pairwise_distances_argmin_min(cluster_centers, embeddings)
        print(closest_indices)
        print([clean_annotation[i] for i in closest_indices])

        # plot the embeddings by TSNE and use color from kmeans
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        import matplotlib.pyplot as plt
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans.labels_)
        plt.savefig('./figs/capture24_label_embedding.png')

        # embeddings = []
        # dataset = Ego4D_Narration(modal=['audio', 'imu'], window_sec=10)
        # for i in tqdm(range(len(dataset))):
        #     data_sample = dataset.window_idx[i]
        #     text = data_sample['text']
        #     embedding = model.encode(text)
        #     embeddings.append(embedding)
        #     # break
        # embeddings = np.stack(embeddings)   
        # np.save('./resources/ego4d_narration_embedding.npy', embeddings)
    elif args.mode == 'cosine':
        capture24 = pd.read_csv('./resources/capture24_label_count.csv')
        clean_annotation = []
        for annotation, count in zip(capture24['annotation'], capture24['count']):
            annotations = annotation.split(';')
            annotation = re.sub(r'\d+', '', annotations[-2])
            clean_annotation.append(annotation)

        imu_attributes = ['walking', 'sitting', 'standing', 'head movement']
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        imu_embeddings = model.encode(imu_attributes)

        capture24_embeddings = np.load('./resources/capture24_label_embedding.npy')
        # tsne = TSNE(n_components=2, random_state=42)
        # embeddings_2d = tsne.fit_transform(capture24_embeddings)

        ego4d_embeddings = np.load('./resources/ego4d_narration_embedding.npy')
        dataset = Ego4D_Narration(modal=['audio', 'imu'], window_sec=10)
        scenario_embeddings = {}
        for i in range(len(dataset)):
            data_sample = dataset.window_idx[i]
            scenario = data_sample['scenario']
            for s in scenario:
                if s not in scenario_embeddings:
                    scenario_embeddings[s] = []
                scenario_embeddings[s].append(i)

        # only keep 50 samples for each scenario
        for s in scenario_embeddings:
            scenario_name = dataset.scenario_map[s].replace('/', '')
            keep_idx = scenario_embeddings[s]

            ego4d_embeddings_keep = ego4d_embeddings[keep_idx]
            cosine_similarity = np.dot(capture24_embeddings, ego4d_embeddings_keep.T)
            # find the index where max cosine similar > 0.5
            max_cosine_sample = np.max(cosine_similarity, axis=0)
            idx_max_cosine_sample = np.where(max_cosine_sample > 0.5)[0]
            # only pick random 50 samples
            idx_max_cosine_sample = np.random.choice(idx_max_cosine_sample, 10)

            for idx in tqdm(idx_max_cosine_sample):
                max_activity = np.argmax(cosine_similarity[:, idx])

                imu_cosine_similarity = np.dot(imu_embeddings, ego4d_embeddings[idx])
                max_imu_activity = np.argmax(imu_cosine_similarity)
                
                activity_description = clean_annotation[max_activity]
                activity_description = activity_description.replace('/', '')

                # break
                data_sample = dataset.__getitem__(idx)
                audio = data_sample['audio']
                imu = data_sample['imu']
                
                audio_name = f'../dataset/ego4d/filter/{idx}_{scenario_name}_{imu_attributes[max_imu_activity]}_{activity_description}.wav'
                imu_name = audio_name.replace('.wav', '.npy')

                np.save(imu_name, imu[:3, ::20].T)
                sf.write(audio_name, audio, 16000)


        # argmax_cosine_sample = np.argmax(cosine_similarity[:50], axis=0)

        # np.save('./resources/ego4d_global_mapping.npy', cosine_similarity[:50])
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 3)
        # # 2D visual of the capture24 embeddings
        # axs[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        # axs[0, 0].scatter(embeddings_2d[:50, 0], embeddings_2d[:50, 1], color='r')
        # axs[0, 0].set_title('embeddings')

        # axs[0, 1].hist(max_cosine_sample, bins=100)
        # axs[0, 1].set_title('cosine similarity')
        # axs[0, 2].bar(range(50), np.bincount(argmax_cosine_sample))
        # axs[0, 2].set_title('number of classes')

        # argmax_cosine_sample = np.argmax(cosine_similarity, axis=0)
        # number_of_classes = np.bincount(argmax_cosine_sample)
        # argmax_class = np.argsort(number_of_classes)[::-1][:50]
        # cosine_similarity = cosine_similarity[argmax_class]
        # np.save('./resources/ego4d_local_mapping.npy', cosine_similarity)

        # axs[1, 0].bar(range(206), number_of_classes)
        # axs[1, 0].bar(argmax_class, number_of_classes[argmax_class], color='r')
        # axs[1, 1].hist(np.max(cosine_similarity, axis=0), bins=100)
        # axs[1, 2].bar(range(50), np.bincount(np.argmax(cosine_similarity, axis=0)))
        # plt.savefig('./figs/ego4d_narration_mapping.png')
    elif args.mode == 'scenario':
    
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
        scenario_names = scenario_names[-10:]

        all_embeddings = []; all_labels = []
        modality = 'audio'
        if modality == 'text':
            # use text embeddings
            ego4d_embeddings = np.load('./resources/ego4d_narration_embedding.npy')
            for s in scenario_names:
                embeddings = ego4d_embeddings[scenario_embeddings[s]]
                all_embeddings.append(embeddings)
                all_labels.append([s]*len(embeddings))
        elif modality == 'audio':
            # use audio embeddings
            import laion_clap
            import librosa
            import torch
            model = laion_clap.CLAP_Module(enable_fusion=False)
            model.load_ckpt() # download the default pretrained checkpoint.
            for s in scenario_names:
                idxs = scenario_embeddings[s]
                idxs = np.random.choice(idxs, 50)
                for idx in idxs:
                    data_sample = dataset.__getitem__(idx)
                    audio = data_sample['audio']
                    audio = librosa.resample(y=audio, orig_sr=16000, target_sr=48000)[None, :]
                    # audio = torch.from_numpy(audio[None, :]).float()
                    audio_embedding = model.get_audio_embedding_from_data(x=audio, use_tensor=False)
                    all_embeddings.append(audio_embedding)
                    all_labels.append([s])
        elif modality == 'imu':
            # use IMU embeddings
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
                    imu = data_sample['imu'][None, :]
                    inputs = {
                        ModalityType.IMU: torch.from_numpy(imu).float().to(device),
                    }
                    with torch.no_grad():
                        embeddings = model(inputs)
                    all_embeddings.append(embeddings[ModalityType.IMU].cpu().numpy())
                    all_labels.append([s])

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        print(all_embeddings.shape, all_labels.shape)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels)
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(f'./figs/ego4d_{modality}_scenario_mapping.png')

    elif args.mode == 'ambient':
        import laion_clap
        import librosa
        import torch
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt() # download the default pretrained checkpoint.

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
        scenario_names = scenario_names[:]

        cosine_similarity = []
        for s in scenario_names:
            idxs = scenario_embeddings[s]
            idxs = np.random.choice(idxs, 10)
            for idx in idxs:
                data_sample = dataset.__getitem__(idx)
                audio = data_sample['audio']
                audio = librosa.resample(y=audio, orig_sr=16000, target_sr=48000)[None, :]
                audio_embedding = model.get_audio_embedding_from_data(x=audio, use_tensor=False)

                text_embedding = model.get_text_embedding(data_sample['text'])
                similarity = np.dot(text_embedding, audio_embedding.T)
                cosine_similarity.append(similarity[0, 0])

        cosine_similarity_home = []
        home_dataset_folder = '../dataset/aiot/Lixing_home-20241106_082431_132'
        audio_files = [f for f in os.listdir(home_dataset_folder) if f.endswith('.mp3')]
        for audio_file in audio_files:
            audio, sr = librosa.load(os.path.join(home_dataset_folder, audio_file), sr=48000)
            audio = audio[None, :]
            audio_embedding = model.get_audio_embedding_from_data(x=audio, use_tensor=False)

            text = audio_file.split(',')[1]
            text_embedding = model.get_text_embedding(text)
            similarity = np.dot(text_embedding, audio_embedding.T)
            cosine_similarity_home.append(similarity[0, 0])

        cosine_similarity = np.abs(cosine_similarity)
        cosine_similarity_home = np.abs(cosine_similarity_home)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        plt.hist(cosine_similarity, bins=100)
        plt.hist(cosine_similarity_home, bins=100, alpha=0.5)
        plt.ylabel('Number of samples')
        plt.xlabel('Cosine similarity')
        plt.savefig('./figs/ego4d_ambient_mapping.png')

