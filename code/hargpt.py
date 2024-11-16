from utils.poe_api import init_poe, call_poe
import numpy as np

def load_capture24():
    capture24_dataset = '../dataset/capture24/'
    data_file = 'P001_X.npy'; label_file = 'P001_Y.npy'
    data = np.load(capture24_dataset + data_file)
    labels = np.load(capture24_dataset + label_file)

    sample_index = 5000;down_sample = 10
    sample = data[sample_index]; label = labels[sample_index]
    sample = sample[::down_sample]
    return sample, label
    

def HARGPT_Template(sample):
    instruction = "You are an expert on analyzing human activities based on IMU recordings."

    content = "The normalized IMU data is collected from a watch attached to the user's wrist with a sampling rate of 10Hz. The IMU data is given in the IMU coordinate frame. The three-axis accelerations are given below: \n"

    content += "x-axis: " + str(sample[:, 0]) + "\n"
    content += "y-axis: " + str(sample[:, 1]) + "\n"
    content += "z-axis: " + str(sample[:, 2]) + "\n"

    content += "The person's action belongs to one of the following categories: [sit-stand, sleep, walking, bicycling]. \
    Could you please tell what action the person was doing based on the given information and IMU readings? Please make an analysis step by step."
    message = instruction + content
    return message

def HeadGPT_Template(sample):
    pass
sample, label = load_capture24()
client, bot, prompt  = init_poe()
message = HARGPT_Template(sample)
response = call_poe(client, bot, prompt, message)    
print(response)
print(label)