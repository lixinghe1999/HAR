import json
import nltk
import matplotlib.pyplot as plt
import gensim.downloader
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# google_news_vectors = gensim.downloader.load('word2vec-google-news-300')
# print('load word2vec')

def separate_noun_verb(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Identify the parts of speech of each word
    pos_tags = nltk.pos_tag(words)

    # Initialize lists to store nouns and verbs
    nouns = []
    verbs = []

    # Iterate through the tagged words and separate nouns and verbs
    for word, tag in pos_tags:
        if tag.startswith('N'):  # Noun tag
            nouns.append(word)
        elif tag.startswith('V'):  # Verb tag
            verbs.append(word)

    return nouns, verbs

def calculate_word_similarity(words1, words2):
    max_sim = 0
    for w1 in words1:
        for w2 in words2:
            # check if the words are in the model's vocabulary
            if w1 not in google_news_vectors or w2 not in google_news_vectors:
                continue
            simi = google_news_vectors.similarity(w1, w2)
            if simi > max_sim:
                max_sim = simi
    return max_sim

json_file = 'qwen_response.json'
with open(json_file) as f:
    data = json.load(f)
cluster_count = {'speech': 0, 'misleading': 0, 'ambiguous':0}
for i, d in enumerate(data):
    n, v = separate_noun_verb(d['text'])
    if 'speaking' in d['result_audio'] or 'talking' in d['result_audio'] or 'saying' in d['result_audio'] or 'speaks' in d['result_audio'] \
    or 'talks' in d['result_audio'] or 'says' in d['result_audio'] or 'singing' in d['result_audio'] or 'sings' in d['result_audio']:
        cluster_count['speech'] += 1
    elif d['cosine_similarity_audio'] < 0.2:
        cluster_count['misleading'] += 1
    else:
        _n, _v = separate_noun_verb(d['result_audio'])
        cluster_count['ambiguous'] += 1
        # print(i, d['text'], d['result_audio'], d['cosine_similarity_audio'])
        # print('Nouns:', n, _n, 'Verbs:', v, _v)
        # print(calculate_word_similarity(n, _n), calculate_word_similarity(v, _v))
    # if i > 50:
    #     break
cluster_count = {k:v/(i+1) for k, v in cluster_count.items()}

print(cluster_count)
plt.bar(cluster_count.keys(), cluster_count.values())
plt.savefig('figs/cluster_count.png')
