import os
import numpy as np
import glob


data_path = '/hdd/1/liyuanming/ComputerVision/dataset/diving/diving/diving_samples_len_151_lstm'
all_videos = glob.glob(data_path + '/*')

print('总视频数：',len(all_videos))