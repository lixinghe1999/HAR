# HAR

## Usage
- ego4d_audio: audio only model for Ego4D (Moments)
- ego4d_imu: IMU only model for Ego4D (Moments)
- egopose_action.py: The egopose subtask of EgoExo4D.
- baseline.py: LIMU_BERT + EfficientAT
- limu_bert.py: The reimplemented LIMU_BERT, no gain compared to the original
- plot.ipynb: This is just a Jupyter Notebook for plotting.

## Folder
- cache: the cache folder for LLM (Qwen2), no need to do anything
- EfficientAT: [EfficientAT](https://github.com/fschmid56/EfficientAT?tab=readme-ov-file) with minor modifications
- ego4d: all the neccessary files for Ego4D
- egoexo: all the neccessary files for EgoExo4D
- figs: all the middle output and visualization, no need to store
- models: IMU-related models
- resources: saved outputs
- utils: remained utilities