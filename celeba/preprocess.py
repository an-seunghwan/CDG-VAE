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
import tqdm
#%%
train = True
img_size = 128 # resize image
causal_structure = 'smile'
# causal_structure = 'attractive'
#%%
base_dir = './CelebAMask-HQ'
img_list = sorted([x for x in os.listdir(base_dir + '/CelebA-HQ-img') if x != '.DS_Store'])
partition = pd.read_csv(base_dir + '/list_eval_partition.txt', sep=' ', header=None)
if train:
    tmp = [x.lstrip('0') for x in partition[partition[1] == 0][0].to_list()]
    img_list = [x for x in img_list if x in tmp]
else:
    tmp = [x.lstrip('0') for x in partition[partition[1] == 2][0].to_list()]
    img_list = [x for x in img_list if x in tmp]
    
anno_base = sorted([x for x in os.listdir(base_dir + '/CelebAMask-HQ-mask-anno') if x != '.DS_Store'])
anno_list = []
for a in anno_base:
    anno_list += os.listdir(base_dir + f'/CelebAMask-HQ-mask-anno/{a}')
with open(base_dir + '/CelebAMask-HQ-attribute-anno.txt', 'r') as f:
    labels = f.readlines()

lables = pd.DataFrame(
    [x.split() for x in labels[2:]],
    columns=['file'] + labels[1].split()
)

if causal_structure == 'smile':
    nodes = ['Smiling', 'Male', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Chubby', 'Narrow_Eyes']
elif causal_structure == 'attractive':
    nodes = ['Young', 'Male', 'Bags_Under_Eyes', 'Chubby', 'Heavy_Makeup', 'Receding_Hairline']
else:
    raise ValueError('Not supported causal structure!')

df_label = lables[['file'] + nodes]
df_label[nodes] = df_label[nodes].astype(float)
df_label[nodes] = df_label[nodes].replace(-1, 0)
df_label = df_label

if causal_structure == 'smile':
    seg_map = [['skin'],  # High_Cheekbones
        ['mouth', 'u_lip', 'l_lip'], # Mouth_Slightly_Open
        ['skin', 'nose', 'neck', 'neck_l'], # Chubby
        ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'], # Narrow_Eyes
        ['l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc
elif causal_structure == 'attractive':
    seg_map = [['l_eye', 'r_eye', 'eye_g'],  # Bags_Under_Eyes
        ['skin', 'nose', 'neck', 'neck_l'], # Chubby
        ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'u_lip', 'l_lip'], # Heavy_Makeup
        ['hair', 'hat'], # Receding_Hairline
        ['mouth', 'l_ear', 'r_ear', 'ear_r', 'cloth', 'hair', 'hat']] # etc
else:
    raise ValueError('Not supported causal structure!')
#%%
#%%
index = 0
tag = list(map(lambda x: 'train' if x else 'test', [train]))[0]
for index in tqdm.tqdm(range(len(img_list)), desc=f"Saving data ({causal_structure} & {tag})..."):
    idx = int(img_list[index].split('.')[0])
    img = cv2.imread(base_dir + '/CelebA-HQ-img/' + img_list[index])
    # b,g,r = cv2.split(img)
    # img = cv2.merge([r,g,b])
    img = cv2.resize(img, (img_size, img_size)) / 255

    b = idx // 2000
    filenames = []
    for seg in seg_map:
        filenames.append([base_dir + f'/CelebAMask-HQ-mask-anno/{b}/' + f'{idx:05d}_{a}.png' for a in seg])

    seg_imgs = []
    for fs in filenames:
        tmp = [cv2.resize(cv2.imread(f), (img_size, img_size)) 
                for f in fs if os.path.exists(f)]
        if len(tmp) != 0:
            tmp = np.concatenate(tmp, axis=-1).sum(axis=-1, keepdims=True)
            tmp[tmp != 0] = 1
        else:
            tmp = np.zeros((img_size, img_size, 1))
        seg_imgs.append(tmp)

    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    concat_img = np.concatenate([img] + seg_imgs, axis=-1)

    y = np.array(df_label[df_label['file'] == img_list[index]].iloc[0, 1:], dtype=np.float32)

    if train:
        np.save(f'./data/train/{causal_structure}/{idx}', concat_img)
        np.save(f'./data/train/label/{idx}', y)
    else:
        np.save(f'./data/test/{causal_structure}/{idx}', concat_img)
        np.save(f'./data/test/label/{idx}', y)
#%%