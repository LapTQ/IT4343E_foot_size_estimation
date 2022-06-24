import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from models.unet import UNet
from utils.dataset import get_dataloader


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--epoch', type=int, default=50)
    ap.add_argument('--train', type=str, default='trainset')
    ap.add_argument('--dev', type=str, default='devset')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--weights', type=str, default=None)

    args = vars(ap.parse_args())

    return args


def main(args):

    train_loader = get_dataloader(
        img_dir=os.path.join(args['train'], 'images'),
        lbl_dir=os.path.join(args['train'], 'labels'),
        batch_size=args['batch_size'],
        shuffle=True
    )

    dev_loader = get_dataloader(
        img_dir=os.path.join(args['dev'], 'images'),
        lbl_dir=os.path.join(args['dev'], 'labels'),
        batch_size=args['batch_size'],
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(3, 2).to(device)

    if args['weights']:
        net.load_state_dict(torch.load(args['weights']))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    for epoch in range(args['epoch']):

        with tqdm(enumerate(train_loader), ascii=True, desc=f'Epoch {epoch}', unit='batch') as t:
            net.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 3 == 2:
                    t.set_postfix(loss=running_loss/3.0)
                    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 3}')
                    last_running_loss = running_loss/3.0
                    running_loss = 0.0

            if not os.path.isdir('weights'):
                os.mkdir('weights')
            torch.save(net.state_dict(), 'weights')

            net.eval()
            total_loss = 0.0
            for i, data in enumerate(dev_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                if i % 3 == 2:
                    t.set_postfix(loss=last_running_loss, val_loss=total_loss/i)


if __name__ == '__main__':

    args = parse_opt()

    main(args)