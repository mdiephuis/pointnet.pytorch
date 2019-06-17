from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)    
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans) # 32, 2500, 3
        x = x.transpose(2, 1) #back transform
        x = F.relu(self.bn1(self.conv1(x))) # after this one 32,64, 2500

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

class Denoiser(nn.Module):
    def __init__(self, feature_transform=False, hidden_dim = 64):
        super(Denoiser, self).__init__()
        self.encoder = PointNetEncoder(feature_transform, hidden_dim)
        self.decoder = PointNetDecoder()
    def forward(self, x):
        x = self.encoder()
        x = self.decoder()



class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform=False, hidden_dim = 64):
        super(PointNetEncoder, self).__init__()
        self.feature_transform=feature_transform
        self.hidden_dim = hidden_dim

        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(self.hidden_dim, 16) #
        self.fc2 = torch.nn.Linear(16, 3)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x) # 32, 3, 2500 -> 32, 1024, 2500
        # x = F.relu(self.bn1(self.conv1(x))) #32, 1088, 2500
        # x = F.relu(self.bn2(self.conv2(x))) # 32, 512, 2500
        # x = F.relu(self.bn3(self.conv3(x))) # 32, 256, 2500
        # x = self.conv4(x) # 32, 128, 2500
        # x = x.transpose(2,1).contiguous() # 32, 2500, 64
        #
        # x = x.view(-1, self.hidden_dim)
        # x = self.fc1(x) # 32*2500 x 64
        # x = self.fc2(x)  # 32 x2500 x 3
        #
        # x = x.view(batchsize, n_pts, 3)


        return x #final size is 32, 64, 2500


class PointNetDecoder(nn.Module):
    """ gets the data with a shape Batch_size, hidden_dims, n_points
    and reduces them back to the original point cloud.
    input size is B_sizex64xnPoints,
    output is B_sizex3xnPoints"""
    def __init__(self, feature_transform=False, k=3):
        super(PointNetDecoder, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(1088, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 3, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(3)

    def forward(self, x):

        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # the input is # 32, 1088, 2500

        x = self.bn1(self.conv1(x)) #32, 32, 2500
        x = self.bn2(self.conv2(x)) # 32, 32, 2500
        x = self.conv3(x) # back to the original size # 32, 3, 2500s


        return x



def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('STN3d')
    print('sim_data stn in ', sim_data.size())
    print('stn out', out.size())
    print('----------------------------------------')
    # print('loss', feature_transform_regularizer(out))

    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('STNkd')
    # print('sim_data64 stn64d in ', sim_data_64d.size())
    # print('stn64d out', out.size())
    # print('----------------------------------------')
    # # print('loss', feature_transform_regularizer(out))
    #
    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('PointNetfeat + pointfeat')
    # print('sim_data global feat in', sim_data.size())
    # print('global feat out', out.size())
    # print('----------------------------------------')
    #
    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('PointNetfeat + pointfeat')
    # print('sim_data point feat in', sim_data.size())
    # print('point feat out', out.size())
    # print('----------------------------------------')
    #
    #
    # cls = PointNetCls(k=5)
    # out, _, _ = cls(sim_data)
    # print('PointNetCls classes')
    # print('class', out.size())
    # print('----------------------------------------')
    #
    #
    # seg = PointNetDenseCls(k=3)
    # x_out, trans_out, trans_feat_out = seg(sim_data)
    # print('PointNetDenseCls')
    # print('sim_data seg in', sim_data.size())
    #
    # print('Output dim, x_out: {}'.format(x_out.size()))
    # print('Check output x')
    # print(x_out[:1, :])
    #
    # if trans_out is not None:
    #     print('Output dim, trans_out: {}'.format(trans_out.size()))
    # else:
    #     print('trans_out was None')
    #
    # if trans_feat_out is not None:
    #     print('Output dim, trans_feat_out: {}'.format(trans_feat_out.size()))
    # else:
    #     print('trans_feat_out was None')
    #
    # print('----------------------------------------')

    encoder = PointNetEncoder()
    x_out = encoder(sim_data)
    decoder = PointNetDecoder()
    x_decoded = decoder(x_out)
    print(x_decoded.shape)
    # print('PointNetDenseEncoder')
    # print('sim_data seg in', sim_data.size())
    # print('Output dim, x_out: {}'.format(x_out.size()))

