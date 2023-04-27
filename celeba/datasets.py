#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os

import torch
#%%
class CelebALoader(torch.utils.data.Dataset):
    def __init__(self, config, train=True):
        
        self.base_dir = './data'
        
        if config["causal_structure"] == 0:
            self.nodes = ['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Chubby', 'Narrow_Eyes']
            if train:
                self.img_dir = self.base_dir + '/train/smile/'
                self.label_dir = self.base_dir + '/train/label/'
            else:
                self.img_dir = self.base_dir + '/test/smile/'
                self.label_dir = self.base_dir + '/test/label/'
                
        elif config["causal_structure"] == 1:
            self.nodes = ['Young', 'Male', 'Bags_Under_Eyes', 'Chubby', 'Heavy_Makeup', 'Receding_Hairline']
            if train:
                self.img_dir = self.base_dir + '/train/attractive/'
                self.label_dir = self.base_dir + '/train/label'
            else:
                self.img_dir = self.base_dir + '/test/attractive/'
                self.label_dir = self.base_dir + '/test/label/'
        else:
            raise ValueError('Not supported causal structure!')
        
        self.img_list = sorted([x for x in os.listdir(self.img_dir) if x != '.DS_Store'])
        self.label_list = sorted([x for x in os.listdir(self.label_dir) if x != '.DS_Store'])
            
    def get_sample(self, index):
        idx = int(self.img_list[index].split('.')[0])
        img = np.load(self.img_dir + f'{idx}.npy')
        y = np.load(self.label_dir + f'{idx}.npy')
        img = torch.from_numpy(img).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        return (img, y)

    def __getitem__(self, index):
        return self.get_sample(index)

    def __len__(self):
        return len(self.img_list)
#%%
# base_dir = './CelebAMask-HQ'
# img_list = sorted([x for x in os.listdir(base_dir + '/CelebA-HQ-img') if x != '.DS_Store'])
# anno_base = sorted([x for x in os.listdir(base_dir + '/CelebAMask-HQ-mask-anno') if x != '.DS_Store'])
# anno_list = []
# for a in anno_base:
#     anno_list += os.listdir(base_dir + f'/CelebAMask-HQ-mask-anno/{a}')
# with open(base_dir + '/CelebAMask-HQ-attribute-anno.txt', 'r') as f:
#     labels = f.readlines()
# #%%
# lables = pd.DataFrame(
#     [x.split() for x in labels[2:]],
#     columns=['file'] + labels[1].split()
# )

# smile = ['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Chubby', 'Narrow_Eyes']
# smile = lables[['file'] + smile]
# smile[['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Chubby', 'Narrow_Eyes']] = smile[['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Chubby', 'Narrow_Eyes']].astype(float)
# smile[smile == -1] = 0

# attractive = ['Young', 'Male', 'Bags_Under_Eyes', 'Chubby', 'Heavy_Makeup', 'Receding_Hairline']
# attractive = lables[['file'] + attractive]
# attractive[['Young', 'Male', 'Bags_Under_Eyes', 'Chubby', 'Heavy_Makeup', 'Receding_Hairline']] = attractive[['Young', 'Male', 'Bags_Under_Eyes', 'Chubby', 'Heavy_Makeup', 'Receding_Hairline']].astype(float)
# attractive[attractive == -1] = 0
# #%%
# atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
#         'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

# smile_seg_map = [['skin'],  # High_Cheekbones
#     ['mouth', 'u_lip', 'l_lip'], # Mouth_Slightly_Open
#     ['skin', 'nose', 'neck', 'neck_l'], # Chubby
#     ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'], # Narrow_Eyes
#     ['l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc

# attractive_seg_map = [['l_eye', 'r_eye', 'eye_g'],  # Bags_Under_Eyes
#     ['skin', 'nose', 'neck', 'neck_l'], # Chubby
#     ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'u_lip', 'l_lip'], # Heavy_Makeup
#     ['hair', 'hat'], # Receding_Hairline
#     ['mouth', 'l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc
# #%%
# tmp = set()
# for x in smile_seg_map:
#     tmp = tmp.union(set(x))
# assert tmp == set(atts)

# tmp = set()
# for x in attractive_seg_map:
#     tmp = tmp.union(set(x))
# assert tmp == set(atts)
# #%%
# img_size = 128

# idx = int(img_list[0].split('.')[0])

# img = cv2.imread(base_dir + '/CelebA-HQ-img/' + img_list[0])
# b,g,r = cv2.split(img)
# img = cv2.merge([r,g,b])
# img = cv2.resize(img, (img_size, img_size)) / 255

# b = idx // 2000
# filenames = []
# for seg in smile_seg_map:
#     filenames.append([base_dir + f'/CelebAMask-HQ-mask-anno/{b}/' + f'{idx:05d}_{a}.png' for a in seg])

# seg_imgs = []
# for fs in filenames:
#     seg_imgs.append(
#         np.concatenate(
#             [cv2.resize(cv2.imread(f), (img_size, img_size)) 
#              for f in fs if os.path.exists(f)], axis=-1
#             ).sum(axis=-1, keepdims=True))

# np.concatenate([img] + seg_imgs, axis=-1)
# #%%
# smile[smile['file'] == img_list[0]].to_numpy()
#%%