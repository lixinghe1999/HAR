import numpy as np
from .features import extract_features
def feature_explain(feature, feature_name):
    return f'{feature_name}: {feature:.2f}'

class track_parser():
    '''
    convert the translation and rotation output to text
    '''
    def __init__(self):
        self.translation_threshold = 0.1
        self.rotation_threshold = 0.1

    def convert(self, translation, rotation):
        '''
        translation: [x, y, z], shape (N, 3)
        rotation: [roll, pitch, yaw], shape (N, 3)
        '''
        distance = translation[-1] - translation[0]
        norm_distance = np.linalg.norm(distance)
        if norm_distance < self.translation_threshold:
            distance_text = 'not moving'
        else:
            # distance_text = f'walked {distance[0]:.2f} meters along x-axis, {distance[1]:.2f} meters along y-axis, {distance[2]:.2f} meters along z-axis'
            distance_text = f'walked {np.linalg.norm(distance):.2f} meters'

        degree = rotation[-1] - rotation[0]
        degree_text = ''
        for degree, axs in zip(degree, ['x', 'y', 'z']):
            if degree < self.rotation_threshold:
                pass
            else:
                degree_text += f'rotated {degree:.2f} radians along {axs}-axis, '
        return distance_text, degree_text

class imu_parser():
    def convert(self, gyro, acc):
        '''
        gyro: [x, y, z], shape (3, N)
        acc: [x, y, z], shape (3, N)
        convert them to description of the data
        '''
        gyro_feature = extract_features(gyro.T, sample_rate=200)
        acc_feature = extract_features(acc.T, sample_rate=200)
        # convert dict to array
        gyro_feature = np.array(list(gyro_feature.values()))
        acc_feature = np.array(list(acc_feature.values()))
        # print('gyro_feature', gyro_feature.shape, 'acc_feature', acc_feature.shape)
        # print('gyro_feature', gyro_feature, 'acc_feature', acc_feature)
        return gyro_feature, acc_feature

