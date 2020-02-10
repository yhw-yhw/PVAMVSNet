import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image


def multi_scale(depth_folder_down4, depth_folder_down8, depth_folder_down16, final_folder, test_list):

    depth_folder = depth_folder_down4
    conf_folder = depth_folder
    output_folder = depth_folder
    depth2_folder = depth_folder_down8
    conf2_folder = depth2_folder
    mask2_folder = depth2_folder
    depth3_folder = depth_folder_down16
    conf3_folder = depth3_folder
    mask3_folder = depth3_folder

    if not os.path.exists(final_folder):
        os.mkdir(final_folder)

    file = open(test_list)
    scans = file.readlines()
    scans = [line.rstrip() for line in scans]

    sum = 0
    ct = 0
    shape_all = 0
    for scan in scans:
        ct = ct + 1
        depth_t = os.path.join(depth_folder, scan)

        depth2_t = os.path.join(depth2_folder, scan)
        depth3_t = os.path.join(depth3_folder, scan)
        conf_t = os.path.join(conf_folder, scan)
        conf2_t = os.path.join(conf2_folder, scan)
        conf3_t = os.path.join(conf3_folder, scan)


        mask2_folder_t = os.path.join(mask2_folder, scan)
        mask2_folder_t = os.path.join(mask2_folder_t, 'mask')

        mask3_folder_t = os.path.join(mask3_folder, scan)
        mask3_folder_t = os.path.join(mask3_folder_t, 'mask')

        output_t = os.path.join(output_folder, 'output')

        final_t = os.path.join(final_folder, scan)

        if not os.path.exists(final_t):
            os.mkdir(final_t)

        if not os.path.exists(output_t):
            os.mkdir(output_t)
        output_t = os.path.join(output_t, scan)
        if not os.path.exists(output_t):
            os.mkdir(output_t)
        sum_t = 0
        shape = 0
        for i in range(49):
            print('process depth '+str(i))
            depth_s = os.path.join(depth_t, 'depth_est')

            final_d = os.path.join(final_t, 'depth_est')
            if not os.path.exists(final_d):
                os.mkdir(final_d)
            final_c = os.path.join(final_t, 'confidence')
            if not os.path.exists(final_c):
                os.mkdir(final_c)

            depth_s = os.path.join(depth_s, '%08d.pfm' % i)

            depth2_s = os.path.join(depth2_t, 'depth_est')
            depth2_s = os.path.join(depth2_s, '%08d.pfm' % i)

            mask2_s = os.path.join(mask2_folder_t, '%08d_final.png' % i)

            depth3_s = os.path.join(depth3_t, 'depth_est')
            depth3_s = os.path.join(depth3_s, '%08d.pfm' % i)
            mask3_s = os.path.join(mask3_folder_t, '%08d_final.png' % i)

            conf_s = os.path.join(conf_t, 'confidence')
            conf_s = os.path.join(conf_s, '%08d.pfm' % i)

            final_d_s = os.path.join(final_d, '%08d.pfm' % i)

            final_c_s = os.path.join(final_c, '%08d.pfm' % i)

            conf2_s = os.path.join(conf2_t, 'confidence')
            conf2_s = os.path.join(conf2_s, '%08d.pfm' % i)

            conf3_s = os.path.join(conf3_t, 'confidence')
            conf3_s = os.path.join(conf3_s, '%08d.pfm' % i)
            
            output_s = os.path.join(output_t, 'error_map_%04d.png' % i)

            mask2_img_t = cv2.imread(mask2_s)

            mask2_img_t = cv2.cvtColor(mask2_img_t, cv2.COLOR_BGR2GRAY)

            mask2_arr = np.array(mask2_img_t)

            mask2_t = mask2_arr > 0

            mask3_img_t = cv2.imread(mask3_s)

            mask3_img_t = cv2.cvtColor(mask3_img_t, cv2.COLOR_BGR2GRAY)

            mask3_arr = np.array(mask3_img_t)

            mask3_arr = cv2.pyrUp(mask3_arr)

            depth = read_pfm(depth_s)[0]

            depth2 = read_pfm(depth2_s)[0]

            depth3 = read_pfm(depth3_s)[0]

            depth3 = cv2.pyrUp(depth3)

            confidence = read_pfm(conf_s)[0]

            confidence2 = read_pfm(conf2_s)[0]

            confidence3 = read_pfm(conf3_s)[0]

            confidence3 = cv2.pyrUp(confidence3)

            depth2 = cv2.pyrUp(depth2)

            zero_depth = np.zeros([8, 400])

            depth2 = np.r_[depth2, zero_depth]

            confidence2 = cv2.pyrUp(confidence2)

            zero_conf = np.zeros([8, 400])

            confidence2 = np.r_[confidence2, zero_conf]

            mask2_arr = cv2.pyrUp(mask2_arr)

            zero_depth = np.zeros([8, 400])

            mask2_arr = np.r_[mask2_arr, zero_depth]

            mask3 = (0.9 < confidence2) & (confidence < 0.5) & (mask2_arr > 0)

            depth[mask3] = depth2[mask3]
            confidence[mask3] = confidence2[mask3]

            save_pfm(final_d_s, depth)

            save_pfm(final_c_s, confidence)

