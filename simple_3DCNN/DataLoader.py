from abc import ABC
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image
import glob
import scipy.io as io

TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 10
SAMPLE_FRAME_NUM = 10

# all_labels
labels_dict = io.loadmat('/hdd/1/liyuanming/ComputerVision/dataset/diving/diving/diving_overall_scores.mat')
labels = [i[0] for i in labels_dict['overall_scores']]

# video names in training and testing sets
train_video_names = list()
test_video_names = list()
with open('./aqa_diving_train_list.txt', 'r') as f:
    all_train_video_roots = f.readlines()
    for path in all_train_video_roots:
        video_name = path.split('/')[-1].split('.')[0]
        train_video_names.append(video_name)
with open('./aqa_diving_test_list.txt', 'r') as f:
    all_test_video_roots = f.readlines()
    for path in all_test_video_roots:
        video_name = path.split('/')[-1].split('.')[0]
        test_video_names.append(video_name)
print('{} videos for training.'.format(len(train_video_names)))
print('{} videos for testing.'.format(len(test_video_names)))

# cutting points
cutting_points = [16, 32, 48, 64, 80, 96, 112, 128, 144]


# dataloader for UNLV Dive dataset
class UNLV_Dive_Data(Dataset):
    def __init__(self, frame_root, video_names, isTrain, num_of_frame, transform=None):
        super(UNLV_Dive_Data, self).__init__()
        self.len = len(video_names)
        self.video_names = video_names
        self.num_of_frame = num_of_frame
        self.tag = None
        if isTrain:
            self.tag = 'train'
        else:
            self.tag = 'test'
        self.frame_root = frame_root + '/' + self.tag
        self.transforms = transform

    def __getitem__(self, index):
        # 'aqa_diving_video_frames/' + 'train\test' + video_name
        specific_frame_root = self.frame_root + '/'
        item_video_name = None
        if self.tag == 'train':
            item_video_name = train_video_names[index]
            specific_frame_root += train_video_names[index] + '/' + train_video_names[index] + '_'
        else:
            item_video_name = test_video_names[index]
            specific_frame_root += test_video_names[index] + '/' + test_video_names[index] + '_'
        label = labels[int(item_video_name) - 1]
        # concat all frames together
        one_clip = None
        # print('numFrame:', self.num_of_frame)
        for i in range(self.num_of_frame):
            frame = Image.open(specific_frame_root + str(i) + '.jpg')
            frame_tensor = self.transforms(frame)  # it should be [3, 112, 112]
            frame_tensor = frame_tensor.unsqueeze(0)  # it should be [1, 3, 112, 112]
            if one_clip is None:
                one_clip = frame_tensor
            else:
                one_clip = torch.cat((one_clip, frame_tensor), 0)
        # finally, one_clip: [144, 3, 112, 112]
        # print('finally, one_clip:', one_clip.shape)

        # split the frames into video clips
        video_clips = torch.split(one_clip, 16, 0)  # (clip1, clip2, ..., clip9), clip_n: [16, 3, 112, 112]
        final_video_clips = None
        for clip in video_clips:
            clip = clip.permute(1, 0, 2, 3)  # [16, 3, 112, 112] -> [3, 16, 112, 112]
            clip = clip.unsqueeze(0)  # [3, 16, 112, 112] -> [1, 3, 16, 112, 112]
            if final_video_clips is None:
                final_video_clips = clip
            else:
                final_video_clips = torch.cat((final_video_clips, clip), 0)
        # finally, final_video_clips: [9, 3, 16, 112, 112]

        return final_video_clips, label

    def __len__(self):
        return self.len


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = UNLV_Dive_Data(
    frame_root='../simple_3DCNN/aqa_diving_video_frames',
    video_names=train_video_names,
    isTrain=True,
    num_of_frame=144,
    transform=transform
)

# divide the dataset into batches
trainset_loader = DataLoader(
    trainset,
    batch_size=6,
    shuffle=True,
    num_workers=0
)

testset = UNLV_Dive_Data(
    frame_root='../simple_3DCNN/aqa_diving_video_frames',
    video_names=test_video_names,
    isTrain=False,
    num_of_frame=144,
    transform=transform
)

# divide the dataset into batches
testset_loader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

if __name__ == '__main__':
    # print(os.listdir('.'))
    # classInd, train_video_name_list, test_video_name_list = get_classInd_and_split('../dataset/UCF101/ucf101_splits',
    #                                                                                '1')
    num = 0
    for i, j in trainset_loader:
        if num == 0:
            print(i.shape)
            print(j.shape)
            print(i.permute(1, 0, 2, 3, 4, 5).shape)
            a = i.permute(1, 0, 2, 3, 4, 5)
            for clip_idx in range(a.shape[0]):
                video_clip = a[clip_idx]
                print(video_clip.shape)
        num += 1
