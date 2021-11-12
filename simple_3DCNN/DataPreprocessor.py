import os
import cv2
import numpy as np
import torch
import random
import glob

def get_video_data(video_path, out_path, num_of_frame=144, is_train=True):
    cap = cv2.VideoCapture(video_path)
    # 获取视频帧数
    numFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.split('/')[-1].split('.')[0]

    # 要为每个视频建立一个文件夹
    if is_train:
        if not os.path.isdir(out_path + '/train/' + str(video_name)):
            print('creating folder: ' + out_path + '/train/' + str(video_name))
            os.makedirs(out_path + '/train/' + str(video_name))
    else:
        if not os.path.isdir(out_path + '/test/' + str(video_name)):
            print('creating folder: ' + out_path + '/test/' + str(video_name))
            os.makedirs(out_path + '/test/' + str(video_name))

    # 随机生成视频帧
    frame_seq = random.sample(range(0, numFrame - 1), num_of_frame)
    frame_idx = 0
    frame_list = list()
    while True:
        # 获取151帧的信息
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_seq:
            frame_list.append(frame)
        frame_idx += 1

    print('Need to save {} frames'.format(len(frame_list)))
    for i in range(len(frame_list)):
        # TODO：把需要存的帧存起来
        if is_train:
            cv2.imwrite(out_path + '/train/' + video_name + '/' + video_name + '_' + str(i) + '.jpg', frame_list[i])
        else:
            cv2.imwrite(out_path + '/test/' + video_name + '/' + video_name + '_' +str(i) + '.jpg', frame_list[i])
    print('video{}, finished.'.format(video_name))
    return

if __name__ == '__main__':
    data_path = '/hdd/1/liyuanming/ComputerVision/dataset/diving/diving/diving_samples_len_151_lstm'
    all_videos = glob.glob(data_path + '/*')
    print('总视频数：', len(all_videos))
    # 生成训练集和测试集的文件名
    test_list = random.sample(all_videos, 70)
    train_list = random.sample(all_videos, 300)
    print('Size of test set:', len(test_list))
    print('Size of train set:', len(train_list))
    with open('./aqa_diving_train_list', 'w') as f:
        for i in train_list:
            f.write(i)
            f.write('\n')
    with open('./aqa_diving_test_list', 'w') as f:
        for i in train_list:
            f.write(i)
            f.write('\n')
    # 存入抽取出来的视频帧
    out_path = './aqa_diving_video_frames'
    for video_path in train_list:
        get_video_data(video_path, out_path, is_train=True)
    for video_path in test_list:
        get_video_data(video_path, out_path, is_train=False)
