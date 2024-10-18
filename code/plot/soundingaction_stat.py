import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def word2vec(words):
    '''
    Convert the words to word vectors
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for word in words:
        embedding = model.encode([word])
        embeddings.append(embedding)
    return np.concatenate(embeddings, axis=0)
def ego4d_taxonomy():
    '''
    Cluster the moments into different categories
    '''

    noun_file = '../dataset/ego4d/v2/annotations/narration_noun_taxonomy.csv'
    verb_file = '../dataset/ego4d/v2/annotations/narration_verb_taxonomy.csv'
    # converter groups, str to list
    noun = pd.read_csv(noun_file, delimiter=',', converters={'group': eval})
    verb = pd.read_csv(verb_file, delimiter=',', converters={'group': eval})
    # convert noun and verb to dict mapping,  two column, label, group, map each of groups to label
    noun_dict = {}; verb_dict = {}
    for idx, row in noun.iterrows():
        for g in row['group']:
            noun_dict[g] = (row['label'].strip(), idx)
    for idx, row in verb.iterrows():
        for g in row['group']:
            verb_dict[g] = (row['label'].strip(), idx)
    return noun_dict, verb_dict, noun['label'], verb['label']

metadata_dir = '../dataset/ego4d/soundingaction/'
meta_file = 'egoclip_val_clean_audio_halo.csv'
metadata = pd.read_csv(metadata_dir + meta_file, header=0, sep='\t')
noun_dict, verb_dict, noun_name, verb_name = ego4d_taxonomy()
print('number of noun:', len(noun_name), 'number of verb:', len(verb_name))
# noun_embeddings = word2vec(noun_name)
# verb_embeddings = word2vec(verb_name)
# np.save('./resources/soundingaction_noun_embedding.npy', noun_embeddings)
# np.save('./resources/soundingaction_verb_embedding.npy', verb_embeddings)
noun_embeddings = np.load('./resources/soundingaction_noun_embedding.npy')
verb_embeddings = np.load('./resources/soundingaction_verb_embedding.npy')  

noun_group = []
verb_group = {verb:[] for verb in verb_name.to_list()}
for idx, row in metadata.iterrows():
    clip_text = row['clip_text']
    noun_group.append([])
    for word in clip_text.split():
        word = word.lower()
        if word.endswith('es'):
            word = word[:-2]
        if word.endswith('s'):
            word = word[:-1]
        if word in verb_dict:
            _verb_group, _ = verb_dict[word]
            verb_group[_verb_group].append(idx)
        if word in noun_dict:
            _noun_group, noun_id = noun_dict[word]
            noun_group[idx].append(noun_id)
postive_idx = metadata[metadata['positive'] == 1].index
positive_ratio = {}
for verb, idxs in verb_group.items():
    _positive_count = sum([1 for idx in idxs if idx in postive_idx])
    if len(idxs) > 0:
        positive_ratio[verb] = _positive_count / len(idxs)
    else:
        positive_ratio[verb] = 0
    if positive_ratio[verb] > 0.5:
        print(verb, positive_ratio[verb], len(idxs))