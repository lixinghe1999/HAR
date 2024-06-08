import numpy as np
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def label_dict(dataset):
    label_dict = {}
    for data in dataset:
        if 'text' not in data:
            data = data[-1]
        label = data['text']
        if label not in label_dict:
            label_dict[label] = len(label_dict) + 1
        else:
            label_dict[label] += 1
    # sort it
    label_dict = dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=True))
    label_dict = {k: i for i, (k, v) in enumerate(label_dict.items())}
    return label_dict  
def cluster_plot(label_dict):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = model.to('cuda')
    model.eval()
    embeddings = []
    for label in tqdm(label_dict):
        sentence_features_avg = model.encode([label])
        embeddings.append(sentence_features_avg)
    embeddings = np.concatenate(embeddings, axis=0)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(embeddings)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], )
    # for i, label in enumerate(label_dict):
    #     plt.annotate(label, (reduced_features[i, 0], reduced_features[i, 1]))
    plt.title('Vector Clusters')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('vector_clusters.png')
def cluster_map(label_dict, file_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = model.to('cuda')
    model.eval()
    embeddings = []
    for label in label_dict:
        sentence_features_avg = model.encode([label])
        embeddings.append(sentence_features_avg)
    embeddings = np.concatenate(embeddings, axis=0)
    corr = np.corrcoef(embeddings)
    corr = (corr + corr.T)/2             
    np.fill_diagonal(corr, 1)

    # cluster if the correlation is larger than 0.9
    from scipy.cluster.hierarchy import linkage, fcluster    
    from scipy.spatial.distance import squareform

    dissimilarity = 1 - np.abs(corr)
    hierarchy = linkage(squareform(dissimilarity), method='average')
    labels = fcluster(hierarchy, 0.7, criterion='distance')

    for i, label in enumerate(label_dict):
        label_dict[label] = int(labels[i]) - 1
    import json
    json.dump(label_dict, open(file_name, 'w'))
    return label_dict, max(labels)
def close_to(label_dict, file_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    target = ["head movement", "stands up", "sits down", "walking"]
    simplify_text = {}
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = model.to('cuda')
    model.eval()
    target_semantic = []
    for t in target:
        target_semantic.append(model.encode([t]))
    target_semantic = np.concatenate(target_semantic, axis=0)
    for i, label in enumerate(label_dict):
        sentence_features_avg = model.encode(label)
        mat = np.dot(sentence_features_avg, target_semantic.T)
        select_text = np.argmax(mat)
        simplify_text[label] = int(select_text)
    import json
    json.dump(simplify_text, open(file_name, 'w'))
    return simplify_text, len(target)