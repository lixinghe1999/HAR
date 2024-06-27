from egoexo.egoexo_dataset import EgoExo_atomic
from ego4d.ego_dataset import Ego4D_Narration
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.body_sound import Body_Sound
import numpy as np
import datetime
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from sentence_transformers import SentenceTransformer



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--start_epoch', type=int, default=-1)
    args = parser.parse_args()

    model = Body_Sound().to('cuda')
    text_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to('cuda')
    if args.log is None:
        log_dir = 'resources/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        print('log_dir: {}'.format(log_dir))
    else:
        log_dir = 'resources/{}'.format(args.log)
        ckpt = torch.load('resources/{}/body_sound_{}.pth'.format(args.log, args.start_epoch))
        model.load_state_dict(ckpt)
        print('successfully loaded ckpt from {}'.format(args.log))
    dataset = Ego4D_Narration(pre_compute_json='resources/ego4d_audio_imu.json', window_sec=4, modal=['audio', 'imu'])
    # dataset.save_json('resources/ego4d_audio_imu.json')
    train_size, test_size = int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    if args.train:
        Epoch = 10
    else:
        Epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for e in range(0, Epoch):
        if args.train:
            pbar = tqdm(train_loader)
            train_loss = 0
            model.train()
            for i, data in enumerate(pbar):
                audio, imu = data['audio'].to('cuda'), data['imu'].to('cuda')
                optimizer.zero_grad()
                audio_output, imu_output = model(audio, imu)
                loss = model.loss(audio_output, imu_output, model.logit_scale)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                average_loss = train_loss / (i + 1)
                pbar.set_description(f'Loss: {round(loss.item(), 4)}({round(average_loss, 4)})')
            train_loss = average_loss
            torch.save(model.state_dict(), 'resources/{}/body_sound_{}.pth'.format(log_dir, e))
            print(f'Epoch {e} train_loss: {train_loss}')
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader)
        audio_acc, imu_acc = 0, 0
        score = []
        text_embeddings = []
        for i, data in enumerate(pbar):
            if i > 10:
                break
            text_embeddings.append(text_model.encode(data['text']))

            audio, imu = data['audio'].to('cuda'), data['imu'].to('cuda')
            audio_output, imu_output = model(audio, imu)
            acc = model.match_eval(audio_output, imu_output)
            audio_acc += acc[0]
            imu_acc += acc[1]
            pbar.set_description(f'audio_acc: {acc[0]} imu_acc: {acc[1]}')
            score.append(model.profile_match(audio_output, imu_output))
        average_audio_acc = round(audio_acc / (i + 1), 3)
        average_imu_acc = round(imu_acc / (i + 1), 3) 
        score = np.concatenate(score, axis=0)
        text_embeddings = np.concatenate(text_embeddings, axis=0)

    # histgram of the diagonal values
    plt.hist(score)
    plt.title('score distribution, mean: {}'.format(np.mean(score)))
    plt.savefig(os.path.join(log_dir, 'score.png'))
    plt.close()

    # 2D visualization of the text embeddings
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2) 
    text_embeddings = tsne.fit_transform(text_embeddings)
    # use color to denote the score
    plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], c=score)
    # plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1])
    plt.title('text embeddings')
    plt.colorbar()
    plt.savefig(os.path.join(log_dir, 'text_embeddings.png'))

    print(f'audio_acc: {average_audio_acc}, imu_acc: {average_imu_acc}')