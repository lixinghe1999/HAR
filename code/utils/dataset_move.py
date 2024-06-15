'''
use the script to move the dataset from the original folder to the processed folder
'''
from_dir = '../dataset/egoexo/takes'
to_dir = '../dataset/egoexo/processed'
import os
from tqdm import tqdm
# types = ['audio_feature.npy', 'audio_tag.json']
# folders = os.listdir(from_dir)
# for folder in tqdm(folders):
#     files = os.listdir(os.path.join(from_dir, folder))
#     os.makedirs(os.path.join(to_dir, folder), exist_ok=True)
#     for file in files:
#         if file in types:
#             new_file = os.path.join(to_dir, folder, file)
#             # move to new
#             os.system('mv {} {}'.format(os.path.join(from_dir, folder, file), new_file))

folders = os.listdir(to_dir)
# remove XXX
for folder in tqdm(folders):
    files = os.listdir(os.path.join(to_dir, folder))
    os.makedirs(os.path.join(to_dir, folder), exist_ok=True)
    for file in files:
        if file == 'tags.json':
            os.system('rm {}'.format(os.path.join(to_dir, folder, file)))