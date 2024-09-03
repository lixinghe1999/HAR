'''
Define the middlewares here
'''
import sys
sys.path.append('middleware/imu')
from middleware.imu.main import LIMU_BERT_Inferencer
from middleware.imu.parser import track_parser, imu_parser
from middleware.audio_imu.filter import Speech_filter

from middleware.audio.spatial import spatial_audio

# imu_middleware = LIMU_BERT_Inferencer(model_cfg='middleware/imu/config/limu_bert_20.json', classifier_cfg='middleware/imu/config/classifier.json',
#                                       ckpt='middleware/imu/0.872_20.pth', device='cuda')
imu_middleware = imu_parser()


sys.path.append('EfficientAT')
from EfficientAT.windowed_inference import EATagger
audio_middleware = EATagger(model_name='dymn10_as', device='cuda')

audio_imu_middleware = Speech_filter()
