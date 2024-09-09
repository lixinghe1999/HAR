import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import metrics
import utils.features as features
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
# For reproducibility
np.random.seed(42)

def RF_classifier(X_train, Y_train, X_test=None, Y_test=None, col_label='annotation'):
    clf = BalancedRandomForestClassifier(
    n_estimators=100,
    replacement=True,
    sampling_strategy='not minority',
    n_jobs=4,
    random_state=42,
    )
    X_train = pd.DataFrame([features.extract_features(x) for x in X_train])
    clf.fit(X_train, Y_train)
    print("Train accuracy:")
    print(metrics.classification_report(Y_train, clf.predict(X_train), zero_division=0))

    if X_test is not None and Y_test is not None:
        X_test = pd.DataFrame([features.extract_features(x) for x in X_test])
        print("Test accuracy:")
        print(metrics.classification_report(Y_test, clf.predict(X_test), zero_division=0))


def extract_windows(data, winsize='10s', col_label='annotation'):
    X, Y = [], []
    for t, w in data.resample(winsize, origin='start'):

        # Check window has no NaNs and is of correct length
        # 10s @ 100Hz = 1000 ticks
        if w.isna().any().any() or len(w) != 1000:
            continue

        x = w[['x', 'y', 'z']].to_numpy()
        y = w[col_label].mode(dropna=False).item()

        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='prepare the dataset', default=False)
    args = parser.parse_args()
    if args.prepare:
        # Load the dataset
        dataset_folder = '/home/lixing/har/dataset/capture24/'
        datas = sorted(os.listdir(dataset_folder))
        datas = [data for data in datas if data.endswith('.csv.gz')]
        anno_label_dict = pd.read_csv(os.path.join(dataset_folder, 'annotation-label-dictionary.csv'), index_col='annotation', dtype='string')
        # Let's load one file
        Xs, Ys = [], []
        for data in datas:
            t_start = time.time()
            user_name = data.split('.')[0]
            print('start loading data', user_name)
            data = pd.read_csv(os.path.join(dataset_folder, data), index_col='time', parse_dates=['time'],
            dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
            X, Y = extract_windows(data, winsize='10s', col_label='annotation')
            # save to improve loading speed
            np.save(dataset_folder + user_name + '_X.npy', X); 
            np.save(dataset_folder + user_name + '_Y.npy', Y)
            Xs.append(X)
            Ys.append(Y)
            print('finish loading data time:', time.time()-t_start)
            # break
        Xs = np.concatenate(Xs)
        Ys = np.concatenate(Ys)
        # convert to the total number of each label
        
    else:
        dataset_folder = 'capture24'
        Xs, Ys = [], []
        for i in tqdm(range(1, 152)):
            # x_name P001_X.npy, y_name P001_Y.npy
            x_name = 'P' + str(i).zfill(3) + '_X.npy'
            y_name = 'P' + str(i).zfill(3) + '_Y.npy'
            Xs.append(np.load('capture24/' + x_name))
            Ys.append(np.load('capture24/' + y_name))
        print('Finish loading data')

        Y_count = pd.Series(np.concatenate(Ys), name='annotation').value_counts()
        # cpa = [anno.split(';')[-2].split()[0] for anno in Y_count.index]
        # met = [anno.split(';')[-1].split()[-1] for anno in Y_count.index]
        # print(cpa)
        # print(met)
        # Y_count['cpa'] = cpa
        # Y_count['met'] = met
        # save as csv
        # Y_count.to_csv('Y_count.csv')
        # we would like to include the top 90 percent of the labels
        topk = 20
        topk_labels = Y_count.index[:topk]
        print(topk_labels)

        plt.bar(range(len(Y_count)), Y_count)
        # the top K with red color
        plt.bar(range(topk), Y_count[:topk], color='red')
        # plot the accumulated distribution (normalized) at second y-axis
        plt.twinx()
        plt.plot(range(len(Y_count)), Y_count.cumsum() / Y_count.sum(), color='green')
        # intersection of the two curves
        plt.plot([topk-1, topk-1], [0, 1], color='b')
        plt.savefig('label_distribution.pdf')
        Y_map = {x: i if x in topk_labels else 'other' for i, x in enumerate(Y_count.index)}
        
        # anno_label_dict = pd.read_csv(os.path.join('/home/lixing/har/dataset/capture24', 'annotation-label-dictionary.csv'), 
        #                               index_col='annotation', dtype='string')
        # # convert anno_label_dict['label:Willetts2018'] to dict
        # Y_map = {k: v for k, v in zip(anno_label_dict.index, anno_label_dict['label:Willetts2018'])}
        # # # select top 80% as training data
        # if len(Xs) == 1:
        #     X_train = np.concatenate(Xs); Y_train = np.concatenate(Ys)
        #     X_test = None; Y_test = None
        #     Y_train = np.array([Y_map[y] for y in Y_train])
        # else:
        #     train_size = int(0.8 * len(Xs))
        #     X_train = np.concatenate(Xs[:train_size]); Y_train = np.concatenate(Ys[:train_size])
        #     X_test = np.concatenate(Xs[train_size:]); Y_test = np.concatenate(Ys[train_size:])

        #     # map the labels to top 10 labels and others
        #     Y_train = np.array([Y_map[y]  for y in Y_train])
        #     Y_test = np.array([Y_map[y]  for y in Y_test])
        # RF_classifier(X_train, Y_train, X_test, Y_test, 'label')
