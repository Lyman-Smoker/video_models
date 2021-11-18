# from video_models.simple_3DCNN.C3D import C3D
# from video_models.simple_3DCNN.AQA_Regressor import Regressor
# from video_models.simple_3DCNN.DataLoader import trainset_loader, testset_loader
from C3D_model import C3D
from AQA_Regressor import Regressor
from DataLoader import trainset_loader, testset_loader
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import torch.nn as nn
import random
import matplotlib as mpl
mpl.use('Agg')
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

writer = SummaryWriter('./runs/')

torch.manual_seed(0); torch.cuda.manual_seed_all(0); random.seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic=True


EPOCH = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9
SAVE_INTERVAL = 300

# 设置运行设备
device_ids = [4, 6]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

# 载入模型
c3d = C3D(num_classes=101, pretrained=True, pretrained_model_path='./c3d-pretrained.pth')
c3d = torch.nn.DataParallel(c3d, device_ids=device_ids)
c3d = c3d.cuda(device=device_ids[0])

# 冻结参数
for param in c3d.parameters():
    param.requires_grad = False

score_regressor = Regressor()
score_regressor = torch.nn.DataParallel(score_regressor, device_ids=device_ids)
score_regressor = score_regressor.cuda(device=device_ids[0])
c3d.train()
score_regressor.train()

for name, param in c3d.named_parameters():
    writer.add_histogram(name + '_c3d', param.clone().data.view(-1, 1).to('cpu').numpy(), 0)
for name, param in score_regressor.named_parameters():
    writer.add_histogram(name + '_rgs', param.clone().data.view(-1, 1).to('cpu').numpy(), 0)


optimizer = optim.SGD(
    params=[{'params': c3d.parameters(), 'lr': LEARNING_RATE * 0.0001} , {'params':score_regressor.parameters(), 'lr': LEARNING_RATE}],
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)


# save checkpoint
def save_checkpoint(path, model, optimizer):
    state = {
        'c3d': model[0].state_dict(),
        'regressor': model[1].state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def train(epoch, save_interval):
    iteration = 0

    # define loss function
    loss = nn.MSELoss()

    for i in range(epoch):
        print('epoch:', i)
        # train
        c3d.train()
        score_regressor.train()
        it = 0
        total_loss = 0
        print('training: ')
        for video_clips, label in tqdm(trainset_loader):
            # put data into GPU
            # video_clips = video_clips.cuda(device=device_ids[0])
            # label = label.unsqueeze(1).type(torch.FloatTensor).cuda(device=device_ids[0])
            # clip_feats = torch.Tensor([]).cuda(device=device_ids[0])

            # run on CPU
            label = label.unsqueeze(1).type(torch.FloatTensor)
            clip_feats = torch.Tensor([])


            video_clips.transpose_(0, 1)

            for video_clip in video_clips:

                video_feature = c3d(video_clip)

                video_feature.unsqueeze_(0)
                video_feature.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, video_feature), 1)
            # clip_feats:[6, 9, 8192]
            video_feature_average = clip_feats.mean(1)
            # video_feature_average:[6, 8192]

            pred_score = score_regressor(video_feature_average)
            final_loss = loss(pred_score, label)

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('./models/checkpoint-%i.pth' % iteration, (c3d, score_regressor), optimizer)

            iteration += 1
            total_loss += final_loss.item()
            # print("Loss: " + str(loss.item()))
            with open('log.txt', 'a') as f:
                f.write("Epoch " + str(i + 1) + ", Iteration " + str(it + 1) + "'s Loss: " + str(final_loss.item()) + "\n")
            it += 1
        print('Loss: ', total_loss/it)
        writer.add_scalar('loss:', total_loss / it, i)
        for name, param in c3d.named_parameters():
            writer.add_histogram(name + '_c3d', param.clone().data.view(-1, 1).to('cpu').numpy(), i+1)
        for name, param in score_regressor.named_parameters():
            writer.add_histogram(name + '_rgs', param.clone().data.view(-1, 1).to('cpu').numpy(), i+1)
        # test
        # print('testing:')
        # it = 0
        # loss_sum = 0
        # c3d.eval()
        # score_regressor.eval()
        #
        # y_gt = list()
        # y_pred = list()
        #
        # for video_clips, label in tqdm(testset_loader):
        #     train_batch_size = label.shape[0]
        #     # put data into GPU
        #     video_clips = video_clips.cuda(device=device_ids[0])
        #     label = label.unsqueeze(1).type(torch.FloatTensor).cuda(device=device_ids[0])
        #     # label = label.unsqueeze(1).type(torch.FloatTensor)
        #     video_feature_sum = torch.zeros((train_batch_size, 8192), dtype=torch.float).cuda(device=device_ids[0])
        #     # video_feature_sum = video_feature_sum.cuda(device=device_ids[0])
        #     video_clips = video_clips.permute(1, 0, 2, 3, 4, 5)
        #     for clip_idx in range(video_clips.shape[0]):
        #         video_clip = video_clips[clip_idx]
        #         # print('video clip.shape = ', video_clip.shape)
        #         video_feature = c3d(video_clip)
        #         # video_feature = video_feature.cuda(device=device_ids[0])
        #         # print('video feature.shape = ', video_feature.shape)
        #         video_feature_sum += video_feature
        #     video_feature_average = video_feature_sum / video_clips.shape[0]
        #     pred_score = score_regressor(video_feature_average)
        #     y_gt.append(label[0].item())
        #     y_pred.append(pred_score[0].item())
        #     # print('{}\t{}'.format(pred_score[0].item(), label[0].item()), end='    ')
        #     final_loss = loss['criterion_final_score'](pred_score, label)
        #     # print('loss: ', final_loss)
        #     loss_sum += final_loss
        #     it += 1
        # print('test loss: ', loss_sum / it)
        # plt.scatter(np.array(y_gt), np.array(y_pred))
        # plt.plot([-1, 100], [-1, 100])
        # plt.savefig('./result_imgs/test_result_epoch' + str(i) + '.png')
        # # plt.show()
        # plt.close()

    # save_checkpoint('./model/checkpoint-%i.pth' % iteration, (c3d, score_regressor), optimizer)
    save_checkpoint('./models/checkpoint-%i.pth' % iteration, (c3d, score_regressor), optimizer)



if __name__ == '__main__':
    train(EPOCH, SAVE_INTERVAL)
