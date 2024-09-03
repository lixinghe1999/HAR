## EarSAVAS
```
python EarVAS_evaluation.py Dataset.dataset_dir=. Model.exp_dir=.  Model.task=feedforward_audio_and_imu Model.device=cuda Model.samosa=False
```

### Data Format
- Audio sample rate: 16000
- IMU sample rate: 100, unit: 'in order to convert the raw accelerometer readings to units of g, a multiplication factor of 0.000122 is required. Similarly, gyroscope requires a multiplication factor of 0.015267'
- Segment length: 1
- Organize as data_dict[user_name][class_type][file_name][segments of data] in pkl file
- Output: ['Blow_Nose', 'Throat_Clear', 'Sniff', 'Sigh', 'Drink', 'Speech', 'Single_Cough', 'Continuous_Cough', 'Chewing', 'others']

## IMU
- IMU sample rate: 20
- Segment length: 2
- Output: []


## Audio