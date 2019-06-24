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
from sklearn.metrics import mean_squared_error
import numpy as np
from show3d_balls import showpoints


sys.path.append("..")

from pointnet.dataset import ShapeNetDataset
from pointnet.encoder_decoder_model import Denoiser, feature_transform_regularizer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=12, help='input batch size')
parser.add_argument('--outf', type=str, default='denoising', help='output folder')
parser.add_argument('--model', type=str, default='denoising/denoising_model_Lamp_5', help='model path')
parser.add_argument('--dataset', type=str, required=False, help="dataset path",
                    default='../scripts/shapenetcore_partanno_segmentation_benchmark_v0/')

parser.add_argument('--class_choice', type=str, default='Lamp', help="class_choice") #changed for None to get random classes
parser.add_argument('--feature_transform', action='store_false', help="use feature transform")
parser.add_argument('--num_points', type = int, default = 8000, help='the  size of the points in a cloud')
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
    num_workers=1)

blue = lambda x: '\033[94m' + x + '\033[0m'

encoder_decoder = Denoiser(feature_transform=False, hidden_dim=64)


if opt.model != '':
    encoder_decoder.load_state_dict(torch.load(opt.model))
else:
    print("Attention, the model is not pre-trained")


for i, data in enumerate(testdataloader, 0):
    points, target = data
    points = points.transpose(2, 1)
    target = target.transpose(2,1)

    if opt.use_cuda:
       points, target = points.cuda(), target.cuda()
    else:
       points, target = points.double(), target.double()

    pred = encoder_decoder(points)
    mse_val = F.mse_loss(pred, target)

    # sklearn mse
    print("MSE between noisy and GT and denoised and GT are: %f and %f" %(mean_squared_error(points.numpy(),target.numpy()),
                                                                          mean_squared_error(pred.numpy(), target.numpy) ))

    showpoints(pred.numpy(), ballradius=3)
    showpoints(target.numpy(), ballradius=3)

