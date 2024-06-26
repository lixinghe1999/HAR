from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.body_sound import Body_Sound
import torch

def post_process(text):
    print('origial text', text)
    text = text.replace('\n', '-')
    text = text.split('-')
    text = [t.strip() for t in text]
    # if len(text) == 1:
    #     text = [text[0], text[0]]
    print('post process text', text)
    return text
# dataset = EgoExo_atomic(pre_compute_json='resources/egoexo_atomic.json', window_sec=4, modal=[])
# word_dict = {}
# for i in tqdm(range(100)):
#     data = dataset[i] 
#     sound = post_process(data['sound'])
#     for s in sound:
#         if s not in word_dict:
#             word_dict[s] = 0
#         word_dict[s] += 1
# print(word_dict)
if __name__ == '__main__':
    dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_audio_imu.json', window_sec=4, modal=['audio', 'imu'])
    # dataset.save_json('resources/ego4d_audio_imu.json')
    train_size, test_size = int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = Body_Sound().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for e in range(1):
        pbar = tqdm(train_loader)
        train_loss = 0
        model.train()
        for i, data in enumerate(pbar):
            audio, imu = data['audio'].to('cuda'), data['imu'].to('cuda')
            optimizer.zero_grad()
            audio_output, imu_output = model(audio, imu)
            loss = torch.nn.functional.mse_loss(audio_output, imu_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            average_loss = train_loss / (i + 1)
            pbar.set_description(f'Loss: {round(loss.item(),3)}({round(average_loss,3)})')
        train_loss = average_loss

        model.eval()
        with torch.no_grad():
            pbar = tqdm(test_loader)
            audio_acc, imu_acc = 0, 0
            for i, data in enumerate(pbar):
                audio, imu = data['audio'].to('cuda'), data['imu'].to('cuda')
                audio_output, imu_output = model(audio, imu)
                acc = model.match_eval(audio_output, imu_output)
                audio_acc += acc[0].item()  
                imu_acc += acc[1].item()
                pbar.set_description(f'audio_acc: {acc[0].item()} imu_acc: {acc[1].item()}')  
            average_audio_acc = round(audio_acc / len(test_loader), 3)
            average_imu_acc = round(imu_acc / len(test_loader), 3)        
        print(f'Epoch {e} train_loss: {train_loss}, audio_acc: {average_audio_acc}, imu_acc: {average_imu_acc}')
        torch.save(model.state_dict(), 'resources/body_sound.pth')