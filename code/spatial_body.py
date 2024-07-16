from egoexo.egoexo_dataset import EgoExo_atomic
from egoexo.aoa import init_music, inference_music
from egoexo.range import calculate_range
import soundfile as sf
import torch

dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic.json', modal=['audio', 'imu'], window_sec=2)
train_idx, test_idx = dataset.split_with_scenario(ratio=0.8)
train_dataset = torch.utils.data.Subset(dataset, train_idx) 
test_dataset = torch.utils.data.Subset(dataset, test_idx)
print(len(train_dataset), len(test_dataset))
# dataset.prune_slience()
# dataset.negative()
# dataset.save_json('resources/egoexo_atomic_negative.json')
# dataset.prune_slience('resources/egoexo_atomic_negative_prune.json')
# dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic_prune.json', modal=['audio', 'imu'], window_sec=2)
# dataset_nagative = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic_negative_prune.json', modal=['audio', 'imu'], window_sec=2)
# print(len(dataset), len(dataset_nagative))
# algo = init_music()
for i in range(len(dataset)):
    data = dataset[i]
    # print(data['text'], data['task_name'], data['scenario'])
#     audio = data['audio']
#     # sf.write('test.wav', audio.T, 16000)
#     r = calculate_range(audio)
#     aoa = inference_music(algo, audio)
#     print(data['text'], r, aoa)
#     if i > 10:
#         break