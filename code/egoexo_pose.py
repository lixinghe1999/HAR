from models.egoexo_models import EgoExo4D_Baseline
from models.egoexo_motion import EgoExo4D_Motion
from egoexo_dataset import EgoExo_pose
import torch
from tqdm import tqdm
def main(model, device, num_epochs=1):
    batch_size = 16
    lr = 1e-5

    train_dataset = EgoExo_pose(split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = EgoExo_pose(split='val', max_frames=5000)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_acc = 100, 100
    log_interval = 10
    test_interval = 500
    for _ in range(num_epochs):
        train_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            target = data['gt'].to(device)
            visible = data['visible'].to(device)
            output = model(data, device)
            loss = model.loss(output, target, visible)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % log_interval == 0:
                writer.add_scalar('Train_loss/train', loss.item(), i)
            if i % test_interval == 0 and i > 0:
                train_loss /= len(train_loader)
                test_error = test(test_dataset, model, device)
                for joint_name in test_error:
                    writer.add_scalar(f'Test_Error/{joint_name}', test_error[joint_name], i)
                if best_loss > train_loss:
                    best_loss = train_loss
                    best_acc = test_error['mean']
                    ckpt_best = model.state_dict()
                    torch.save(model.state_dict(), log_dir + '/bodypose_{}_{}.pth'.format(i, round(test_error['mean'], 3)))
    torch.save(ckpt_best, log_dir + '/bodypose_best.pth')
    print(f"Best Loss: {best_loss}, Best Accuracy: {best_acc}")
    writer.close()

def test(dataset, model, device):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    test_error = {}
    for joint_name in model.joint_names:
        test_error[joint_name] = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            target = data['gt'].to(device)
            visible = data['visible'].to(device)
            output = model(data, device)
            model.evaluate(output, target, visible, test_error)
    for joint_name in model.joint_names:
        test_error[joint_name] = sum(test_error[joint_name])/len(test_error[joint_name])
    mean_error = sum(test_error.values())/len(test_error)
    test_error['mean'] = mean_error
    return test_error


if __name__ == "__main__":
    # Define the model
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    args = parser.parse_args()

    device = 'cuda'
    model = EgoExo4D_Motion().to(device)

    if args.train:
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/" + time_now
        writer = SummaryWriter(log_dir=log_dir)
        main(model, device)
    else:
        ckpt = torch.load(os.path.join('logs', '20240531-115609', 'bodypose_best.pth'))
        model.load_state_dict(ckpt)
        test_dataset = EgoExo_pose(split='val', max_frames=10000)
        test_error = test(test_dataset, model, device)
        print(test_error)
