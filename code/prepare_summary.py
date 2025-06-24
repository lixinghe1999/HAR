from ego4d.ego4d_dataset import Ego4D_Narration, Ego4D_Action2Sound
import json
from tqdm import tqdm
import os
import soundfile as sf
from sentence_transformers import SentenceTransformer
class MotionCandidate:
    def __init__(self,):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.motion_candidates = ['walking', 'standing', 'playing sports', 'sleeping', 'sitting', 'taking car']
        self.embeddings = self.model.encode(self.motion_candidates)
    def __call__(self, text):
        text_embedding = self.model.encode([text])
        scores = text_embedding @ self.embeddings.T
        return self.motion_candidates[scores.argmax()]
    
def raw_summary():
    motion_candidate = MotionCandidate()
    from code.windowed_inference import EATagger
    model = EATagger(model_name='mn10_as', device='cuda:1')
    dataset = Ego4D_Narration(window_sec=None, modal=['audio', 'imu', 'summary']) 
    print(len(dataset), len(dataset.narration_summary))
    json_path = './resources/ego4d_summary_efficientAT.json'
    outputs = []
    for i in tqdm(range(len(dataset))):
        narrations = dataset.narration_summary[i]
        if len(narrations) == 0:
            continue
        motions = []
        for narration in narrations:
            narration, _, _ = narration
            motions.append(motion_candidate(narration))

        data = dataset[i]
        summary = data['text']
        scenario = data['scenario_name']

        audio = data['audio']
        audio_name = './tmp.wav'
        sf.write(audio_name, audio, 16000)
        # tag the audio file
        tags = model.tag_audio_window('./tmp.wav', window_size=10, hop_length=10)    
        # for each window, print the top 5 tags and their probabilities
        preds = []
        for window in tags:
            preds.append(window['tags'])

        output = {
            'summary': summary,
            'scenario': scenario,
            'motions': motions,
            'audio_tags': preds, 
            'narrations': narrations,
        }
        outputs.append(output)

    os.remove(audio_name)
    with open(json_path, 'w') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

def activity_summary():
    # test_dataset = Ego4D_Action2Sound(window_sec=10, modal=['audio', 'imu', 'summary'], csv='ego4d/soundingaction/egoclip_val_clean_audio_manual.csv')
    train_dataset = Ego4D_Action2Sound(window_sec=10, modal=['audio', 'summary'], csv='ego4d/soundingaction/egoclip_val_clean_audio_halo.csv')
    dataset = train_dataset
    json_path = './resources/ego4d_summary_soundingaction.json'
    outputs = []
    for i in tqdm(range(len(dataset))):
        narrations = dataset.narration_summary[i]
        if len(narrations) == 0:
            continue
        else:
            new_narrations = {'action': [], 'background': []}
            for narration in narrations:
                narration, _, positive = narration
                if positive == 1:
                    new_narrations['action'].append(narration)
                else:
                    new_narrations['background'].append(narration)
            narrations = new_narrations
        data = dataset[i]
        summary = data['text']
        scenario = data['scenario_name']
        output = {
            'summary': summary,
            'scenario': scenario,
            'narrations': narrations,
        }
        outputs.append(output)
    with open(json_path, 'w') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
if __name__ == '__main__':
    # raw_summary()
    activity_summary()
