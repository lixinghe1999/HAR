'''
use the script to move the dataset from the original folder to the processed folder
'''
from_dir = '../dataset/egoexo/processed'
to_dir = '../dataset/egoexo/takes'
import os
from tqdm import tqdm
# types = ['flac', 'npy', 'json']
types = ['']
folders = os.listdir(from_dir)
for folder in tqdm(folders):
    files = os.listdir(os.path.join(from_dir, folder))
    os.makedirs(os.path.join(to_dir, folder), exist_ok=True)
    for file in files:
        # os.system('du -sh {}'.format(os.path.join(from_dir, folder, file)))
        format = file.split('.')[-1]
        if format not in types:
            continue
        new_file = os.path.join(to_dir, folder, file)
        print(new_file)
        # move to new
        os.system('mv {} {}'.format(os.path.join(from_dir, folder, file), new_file))
       