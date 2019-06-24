from __future__ import print_function
import sys
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from logger import Logger

sys.path.append("..")

from pointnet.dataset import ShapeNetDataset
from pointnet.encoder_decoder_model import Denoiser, feature_transform_regularizer


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=12, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument(
    '--nepoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='denoising', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=False, help="dataset path",
                    default='../scripts/shapenetcore_partanno_segmentation_benchmark_v0/')

parser.add_argument('--class_choice', type=str, default=None, help="class_choice") #changed for None to get random classes
parser.add_argument('--feature_transform', action='store_false', help="use feature transform")
parser.add_argument('--num_points', type = int, default = 8000, help='the  size of the points in a cloud')
parser.add_argument('--log-dir', type = str, default ='./logs')
# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: True')


opt = parser.parse_args()

opt.use_cuda = not opt.no_cuda and torch.cuda.is_available()

print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice], denoising =True, npoints=opt.num_points)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False, denoising = True, npoints= opt.num_points)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

encoder_decoder = Denoiser(feature_transform=False, hidden_dim=64)


if opt.model != '':
    encoder_decoder.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(encoder_decoder.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


if opt.use_cuda:
    encoder_decoder.cuda()
else:
    encoder_decoder.double()

num_batch = len(dataset) / opt.batchSize
logger = Logger(opt.log_dir)


for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        target = target.transpose(2,1)

        if opt.use_cuda:
            points, target = points.cuda(), target.cuda()
        else:
            points, target = points.double(), target.double()

        optimizer.zero_grad()
        encoder_decoder = encoder_decoder.train()
        pred = encoder_decoder(points)
        loss = F.mse_loss(pred, target)
        #if opt.feature_transform:
        #    loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        # tboard
        # 1. Log scalar values (scalar summary)
        info = {'training loss': loss.item()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch * num_batch +i)

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            target = target.transpose(2, 1)

            if opt.use_cuda:
                points, target = points.cuda(), target.cuda()
            else:
                points, target = points.double(), target.double()

            encoder_decoder = encoder_decoder.eval()
            pred = encoder_decoder(points)
            loss = F.mse_loss(pred, target)

            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
            # 1. Log scalar values (scalar summary)
            info = {'testing loss': loss.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch * num_batch +i)

    torch.save(encoder_decoder.state_dict(), '%s/denoising_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

