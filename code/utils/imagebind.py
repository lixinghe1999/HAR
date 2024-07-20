import torch
import sys
sys.path.append('./')
from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from tqdm import tqdm
def imagebind_dataset(dataset):
    device = 'cuda'
    imagebind = init_imagebind(device)
    batch_size = 32
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, data in enumerate(tqdm(dataset_loader)):
        embeddings = inference_imagebind(imagebind, data['text'], [(audio, dataset.sr_audio) for audio in data['audio']], None, None, device)
        cosine_audio_text = torch.nn.functional.cosine_similarity(embeddings['audio'], embeddings['text'], dim=1)
        cosine_audio_text = cosine_audio_text.cpu().numpy()
        dataset.add(range(i * batch_size, i * batch_size + len(cosine_audio_text)), 'cosine', cosine_audio_text)
    return dataset
def init_imagebind(device):
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model
def inference_imagebind(model, text_list, audio_paths, imu_paths, image_paths, reference_modal='text', device='cuda'):
    inputs = {
        ModalityType.TEXT: imagebind_data.load_and_transform_text(text_list, device),
        ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(audio_paths, device),
        ModalityType.IMU: imagebind_data.load_and_transform_imu_data(imu_paths, device),
        ModalityType.VISION: imagebind_data.load_and_transform_vision_data(image_paths, device),

    }
    with torch.no_grad():
        embeddings = model(inputs)
    return embeddings

def compare_embeddings(embeddings, reference_modal='text'):
    modalities = list(embeddings.keys())
    modalities.remove(reference_modal)
    output = {}
    for m in modalities:
        # softmax classifier
        # output[m] = torch.softmax(embeddings[reference_modal] @ embeddings[m].T, dim=0)
        # cosine similarity
        output[m] = torch.nn.functional.cosine_similarity(embeddings[reference_modal], embeddings[m], dim=1)
    return output
