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

sys.path.append("..")

from pointnet.dataset import ShapeNetDataset
from pointnet.encoder_decoder_model import Denoiser, feature_transform_regularizer


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=False, help="dataset path",
                    default='../scripts/shapenetcore_partanno_segmentation_benchmark_v0/')

parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_false', help="use feature transform")

# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables cuda (default: False')


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
    class_choice=[opt.class_choice], denoising =True)

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
    data_augmentation=False, denoising = True)
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

optimizer = optim.Adam(encoder_decoder.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


if opt.use_cuda:
    encoder_decoder.cuda()
else:
    encoder_decoder.double()

num_batch = len(dataset) / opt.batchSize

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
        #print(pred.size(), target.size())
        loss = F.mse_loss(pred, target)
        #if opt.feature_transform:
        #    loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        if i % 20 == 0:
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

    torch.save(encoder_decoder.state_dict(), '%s/denoising_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

# ## benchmark mIOU
# shape_ious = []
# for i,data in tqdm(enumerate(testdataloader, 0)):
#     points, target = data
#     points = points.transpose(2, 1)
#
#     if opt.use_cuda:
#         points, target = points.cuda(), target.cuda()
#
#     encoder_decoder = encoder_decoder.eval()
#     pred, _, _ = encoder_decoder(points)
#     pred_choice = pred.data.max(2)[1]
#
#     pred_np = pred_choice.cpu().data.numpy()
#     target_np = target.cpu().data.numpy() - 1
#
#     for shape_idx in range(target_np.shape[0]):
#         parts = range(num_classes)#np.unique(target_np[shape_idx])
#         part_ious = []
#         for part in parts:
#             I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             if U == 0:
#                 iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             part_ious.append(iou)
#         shape_ious.append(np.mean(part_ious))
#
# print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))