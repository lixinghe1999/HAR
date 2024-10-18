from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Sound, Ego4D_Narration_Sequence, Ego4D_Free
import soundfile as sf
import pandas as pd
import re
import numpy as np
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
        from sklearn.manifold import TSNE

        capture24_embeddings = np.load('./resources/capture24_label_embedding.npy')
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(capture24_embeddings)

        ego4d_embeddings = np.load('./resources/ego4d_narration_embedding.npy')
        cosine_similarity = np.dot(capture24_embeddings, ego4d_embeddings.T)
        max_cosine_sample = np.max(cosine_similarity[:50], axis=0)
        argmax_cosine_sample = np.argmax(cosine_similarity[:50], axis=0)

        np.save('./resources/ego4d_global_mapping.npy', cosine_similarity[:50])
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3)
        # 2D visual of the capture24 embeddings
        axs[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        axs[0, 0].scatter(embeddings_2d[:50, 0], embeddings_2d[:50, 1], color='r')
        axs[0, 0].set_title('embeddings')

        axs[0, 1].hist(max_cosine_sample, bins=100)
        axs[0, 1].set_title('cosine similarity')
        axs[0, 2].bar(range(50), np.bincount(argmax_cosine_sample))
        axs[0, 2].set_title('number of classes')

        argmax_cosine_sample = np.argmax(cosine_similarity, axis=0)
        number_of_classes = np.bincount(argmax_cosine_sample)
        argmax_class = np.argsort(number_of_classes)[::-1][:50]
        cosine_similarity = cosine_similarity[argmax_class]
        np.save('./resources/ego4d_local_mapping.npy', cosine_similarity)

        axs[1, 0].bar(range(206), number_of_classes)
        axs[1, 0].bar(argmax_class, number_of_classes[argmax_class], color='r')
        axs[1, 1].hist(np.max(cosine_similarity, axis=0), bins=100)
        axs[1, 2].bar(range(50), np.bincount(np.argmax(cosine_similarity, axis=0)))
        plt.savefig('./figs/ego4d_narration_mapping.png')
    else:
        import torchmetrics
        import torch
        import matplotlib.pyplot as plt

        acc_classwises = []
        fig, axs = plt.subplots(1, 3)
        for i, folder in enumerate(['resources/activity_audio', 'resources/activity_imu']):
            output = '{}/preds.npy'.format(folder)
            gt = '{}/gts.npy'.format(folder)
            preds = np.load(output); gts = np.load(gt)
            accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=50, average=None)
            acc_classwise = accuracy(torch.tensor(preds), torch.tensor(gts)).numpy()
            acc_full = torchmetrics.Accuracy(task='multiclass', num_classes=50)(torch.tensor(preds), torch.tensor(gts)).item()
            acc_classwises.append(acc_classwise)
            class_count = np.bincount(gts)
            axs[0].bar(range(i, 150, 3), acc_classwise, width=1, label=folder.split('/')[-1])
            print(np.mean(acc_classwise), acc_full)
        axs[0].set_xticks([])
        axs[0].legend()
        modality_gap = acc_classwises[0] - acc_classwises[1]
        axs[1].bar(range(50), modality_gap)
        axs[1].set_title('Modality gap')
        
        output = 'resources/activity/preds.npy'
        gt = 'resources/activity/gts.npy'
        preds = np.load(output); gts = np.load(gt)
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=50, average=None)
        acc_classwise = accuracy(torch.tensor(preds), torch.tensor(gts)).numpy()
        axs[2].bar(range(50), acc_classwise)
        plt.savefig('figs/activity_classwise.png')