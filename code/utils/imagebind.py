import torch
import sys
sys.path.append('./')
from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
def init_imagebind(device):
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model
def inference_imagebind(model, text_list, audio_paths, imu_paths, device):
    inputs = {
        ModalityType.TEXT: imagebind_data.load_and_transform_text(text_list, device),
        ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(audio_paths, device),
        ModalityType.IMU: imagebind_data.load_and_transform_imu_data(imu_paths, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)
    #cross_similarity = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
    # print("Audio x Text: ", cross_similarity)
    cross_similarity_AT = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
    cross_similarity_IT = torch.softmax(embeddings[ModalityType.IMU] @ embeddings[ModalityType.TEXT].T, dim=-1)
    return cross_similarity_AT, cross_similarity_IT
