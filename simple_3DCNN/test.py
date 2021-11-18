from C3D_model import C3D
from AQA_Regressor import Regressor
from DataLoader import trainset_loader, testset_loader
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib as mpl
mpl.use('Agg')
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def load_model(model_path):
    model = torch.load(model_path)
    c3d = C3D(num_classes=101)
    c3d.load_state_dict({k.replace('module.', ''):v for k,v in model['c3d'].items()})
    c3d.eval()
    regressor = Regressor()
    regressor.load_state_dict({k.replace('module.', ''):v for k,v in model['regressor'].items()})
    regressor.eval()
    return c3d, regressor

 # define loss funtion
loss = {}
criterion_final_score = nn.MSELoss()
penalty_final_score = nn.L1Loss()
loss['criterion_final_score'] = criterion_final_score
loss['penalty_final_score'] = penalty_final_score

# 设置运行设备
device_ids = [4, 6]
checkpoint = 'checkpoint-300'
c3d, regressor = load_model('./models/'+ checkpoint + '.pth')
c3d = torch.nn.DataParallel(c3d, device_ids=device_ids)
c3d = c3d.cuda(device_ids[0])
regressor = torch.nn.DataParallel(regressor, device_ids=device_ids)
regressor = regressor.cuda(device_ids[0])
y_gt = list()
y_pred = list()
with torch.no_grad():
    for video_clips, label in testset_loader:
        train_batch_size = label.shape[0]
        # put data into GPU
        video_clips = video_clips.cuda(device=device_ids[0])
        label = label.unsqueeze(1).type(torch.FloatTensor).cuda(device=device_ids[0])
        # label = label.unsqueeze(1).type(torch.FloatTensor)
        video_feature_sum = torch.zeros((train_batch_size, 8192), dtype=torch.float)
        video_feature_sum = video_feature_sum.cuda(device=device_ids[0])
        video_clips = video_clips.permute(1, 0, 2, 3, 4, 5)
        for clip_idx in range(video_clips.shape[0]):
            video_clip = video_clips[clip_idx]
            # print('video clip.shape = ', video_clip.shape)
            video_feature = c3d(video_clip)
            video_feature = video_feature.cuda(device=device_ids[0])
            # print('video feature.shape = ', video_feature.shape)
            video_feature_sum += video_feature
        video_feature_average = video_feature_sum / video_clips.shape[0]
        pred_score = regressor(video_feature_average)
        y_gt.append(label[0].item())
        y_pred.append(pred_score[0].item())
        print('pred_score:',pred_score)
        print('label', label)
        # print('{}\t{}'.format(pred_score.item(), label.item()), end='    ')
        final_loss = nn.MSELoss()(pred_score, label)
        print('loss: ',final_loss)

    # print(y_gt)
    # print(y_pred)
    # print(np.array(y_gt))
    # print(np.array(y_pred))
    plt.scatter(np.array(y_gt), np.array(y_pred))
    plt.plot([-1, 100], [-1, 100])
    plt.savefig('test_result_' + checkpoint + '.png')
    plt.show()
