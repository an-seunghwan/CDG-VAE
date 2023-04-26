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
    def __init__(self, config):
        
        self.img_size = config["img_size"]
        
        self.base_dir = './CelebAMask-HQ'
        self.img_list = sorted([x for x in os.listdir(self.base_dir + '/CelebA-HQ-img') if x != '.DS_Store'])
        anno_base = sorted([x for x in os.listdir(self.base_dir + '/CelebAMask-HQ-mask-anno') if x != '.DS_Store'])
        anno_list = []
        for a in anno_base:
            anno_list += os.listdir(self.base_dir + f'/CelebAMask-HQ-mask-anno/{a}')
        with open(self.base_dir + '/CelebAMask-HQ-attribute-anno.txt', 'r') as f:
            labels = f.readlines()
        
        lables = pd.DataFrame(
            [x.split() for x in labels[2:]],
            columns=['file'] + labels[1].split()
        )

        if config["causal_structure"] == 0:
            self.nodes = ['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Chubby', 'Narrow_Eyes']
        elif config["causal_structure"] == 1:
            self.nodes = ['Young', 'Male', 'Bags_Under_Eyes', 'Chubby', 'Heavy_Makeup', 'Receding_Hairline']
        else:
            raise ValueError('Not supported causal structure!')
        
        df_label = lables[['file'] + self.nodes]
        df_label[self.nodes] = df_label[self.nodes].astype(float)
        df_label[self.nodes] = df_label[self.nodes].replace(-1, 0)
        self.df_label = df_label
        
        if config["causal_structure"] == 0:
            self.seg_map = [['skin'],  # High_Cheekbones
                ['mouth', 'u_lip', 'l_lip'], # Mouth_Slightly_Open
                ['skin', 'nose', 'neck', 'neck_l'], # Chubby
                ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'], # Narrow_Eyes
                ['l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc
        elif config["causal_structure"] == 1:
            self.seg_map = [['l_eye', 'r_eye', 'eye_g'],  # Bags_Under_Eyes
                ['skin', 'nose', 'neck', 'neck_l'], # Chubby
                ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'u_lip', 'l_lip'], # Heavy_Makeup
                ['hair', 'hat'], # Receding_Hairline
                ['mouth', 'l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc
        else:
            raise ValueError('Not supported causal structure!')

    def get_sample(self, index):
        idx = int(self.img_list[index].split('.')[0])
        img = cv2.imread(self.base_dir + '/CelebA-HQ-img/' + self.img_list[index])
        # b,g,r = cv2.split(img)
        # img = cv2.merge([r,g,b])
        img = cv2.resize(img, (self.img_size, self.img_size)) / 255
        
        b = idx // 2000
        filenames = []
        for seg in self.seg_map:
            filenames.append([self.base_dir + f'/CelebAMask-HQ-mask-anno/{b}/' + f'{idx:05d}_{a}.png' for a in seg])

        seg_imgs = []
        for fs in filenames:
            tmp = [cv2.resize(cv2.imread(f), (self.img_size, self.img_size)) 
                    for f in fs if os.path.exists(f)]
            if len(tmp) != 0:
                tmp = np.concatenate(tmp, axis=-1).sum(axis=-1, keepdims=True)
                tmp[tmp != 0] = 1
            else:
                tmp = np.zeros((self.img_size, self.img_size, 1))
            seg_imgs.append(tmp)
        
        img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        concat_img = torch.from_numpy(np.concatenate([img] + seg_imgs, axis=-1)).to(torch.float32)
        
        y = np.array(self.df_label[self.df_label['file'] == self.img_list[index]].iloc[0, 1:], dtype=np.float32)
        y = torch.from_numpy(y).to(torch.float32)
        return (concat_img, y)

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