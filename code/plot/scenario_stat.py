from ego4d.ego4d_dataset import Ego4D_Narration

dataset = Ego4D_Narration(folder='../dataset/ego4d/v2/', modal=[])
print(dataset.scenario_map)