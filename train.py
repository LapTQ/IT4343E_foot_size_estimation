import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

import albumentations as A
import cv2

from models.unet import UNet
from utils.dataset import get_dataloader


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--train', type=str, default='trainset')
    ap.add_argument('--dev', type=str, default='devset')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-1)
    ap.add_argument('--weights', type=str, default=None)
    ap.add_argument('--size', type=int, default=224)

    args = vars(ap.parse_args())

    return args


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # net = UNet(3, 3).to(device)
    net = deeplabv3_mobilenet_v3_large(pretrained=True)
    net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=1).to(device)

    if args['weights']:
        print('Loading pretrained at ' + args['weights'])
        net.load_state_dict(torch.load(args['weights']), map_location=device)

    # TODO auto detect #channels
    out_size = net(torch.zeros((1, 3, args['size'], args['size']),
                               dtype=torch.float32).to(device)
                   ).shape[-1]

    transform = A.Compose([
        A.ISONoise(p=0.5),
        # A.GridDropout(p=0.25),
        A.MotionBlur(blur_limit=(3, 10), p=0.5),
        A.SafeRotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        # A.RandomBrightnessContrast(p=1),
    ])

    train_loader = get_dataloader(
        img_dir=os.path.join(args['train'], 'images'),
        lbl_dir=os.path.join(args['train'], 'labels'),
        batch_size=args['batch_size'],
        in_size=args['size'],
        out_size=out_size,
        transform=transform,
        shuffle=True
    )

    dev_loader = get_dataloader(
        img_dir=os.path.join(args['dev'], 'images'),
        lbl_dir=os.path.join(args['dev'], 'labels'),
        batch_size=args['batch_size'],
        in_size=args['size'],
        out_size=out_size,
        transform=None,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    for epoch in range(args['epoch']):

        with tqdm(enumerate(train_loader), ascii=True, desc=f'Epoch {epoch + 1} [{len(train_loader)}]', unit=' batch') as t:
            net.train()
            running_loss = 0.0
            total_loss = 0.0
            for i, data in t:
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # running_loss += loss.item()
                running_loss += float(loss)
                total_loss += float(loss)
                if i % 3 == 2:
                    t.set_postfix(loss=running_loss/3.0)
                    running_loss = 0.0

            total_loss /= len(train_loader)

            if not os.path.isdir('weights'):
                os.mkdir('weights')
            torch.save(net.state_dict(), 'weights/ckpt.pth')

            net.eval()
            total_dev_loss = 0.0
            for data in dev_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_dev_loss += float(loss)

            total_dev_loss /= len(dev_loader)
            t.set_postfix(loss=total_loss, dev_loss=total_dev_loss, lr=optimizer.param_groups[0]['lr'])
            print(optimizer.param_groups[0]['lr'], total_loss, total_dev_loss)
            scheduler.step(total_loss)



if __name__ == '__main__':

    args = parse_opt()

    main(args)


# nn.BCELoss: