#%%
"""
Reference
[1]: https://github.com/huawei-noah/trustworthyAI/blob/master/research/CausalVAE/causal_data/pendulum.py
"""
#%%
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import math
import numpy as np
import pandas as pd 
import tqdm
import warnings
warnings.filterwarnings('ignore')
#%%
if not os.path.exists('./causal_data/pendulum/'): 
    os.makedirs('./causal_data/pendulum/train/')
    os.makedirs('./causal_data/pendulum/test/')
#%%
count = 0
train = pd.DataFrame(columns=['light', 'angle', 'length', 'position'])
test = pd.DataFrame(columns=['light', 'angle', 'length', 'position'])
#%%
"""Addictive Noise Data Generating Process"""
np.random.seed(1)
varphi_list = np.random.normal(scale=0.5, size=100)
theta_list = np.random.normal(scale=0.5, size=100)
center = (10, 10.5)
threshold = 45
l = 8
scale = 0.1
varphi = varphi_list[0]
theta = theta_list[0]
for varphi in tqdm.tqdm(varphi_list):
    for theta in theta_list:
        objects = []
        
        varphi_ = varphi
        theta_ = theta
        # varphi_ = varphi + np.random.normal(scale=0.1) 
        # theta_ = theta + np.random.normal(scale=0.1)
        
        light = center[0] - center[1] * math.tan(math.radians(varphi_ * threshold))
        
        ball = (center[0] + l * math.sin(math.radians(theta_ * threshold)),
                center[1] - l * math.cos(math.radians(theta_ * threshold)))
        ball_x = center[0] + (l + 1.5) * math.sin(math.radians(theta_ * threshold))
        ball_y = center[0] - (l + 1.5) * math.cos(math.radians(theta_ * threshold))

        tan_phi = light - ball[0]
        tan_phi /= 20.5 - ball[1]
        shadow_start = center[0] + center[1] * math.tan(math.radians(varphi_ * threshold))

        length = light - 20.5 * tan_phi - shadow_start + np.random.normal(scale=scale)
        position = shadow_start + length / 2 + np.random.normal(scale=scale)
        
        objects.append(('light', varphi_))
        objects.append(('theta', theta_))
        objects.append(('length', length))
        objects.append(('position', position))
        # name = '_'.join([str(int(y)) for x,y in objects])
        name = '_'.join([str(round(y, 2)) for x,y in objects])
        
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
                
        gun = plt.Polygon(([10, 10.5], ball), color = 'black', linewidth = 3)
        pendulum = plt.Circle(ball, 1.5, color = 'firebrick')
        sun = plt.Circle((light, 20.5), 3, color = 'orange')
        shadow = plt.Polygon(([shadow_start, -0.5], [shadow_start + length, -0.5]), color = 'black', linewidth = 4)
        
        ax = plt.gca()
        ax.add_artist(sun)
        ax.add_artist(gun)
        ax.add_artist(pendulum)
        ax.add_artist(shadow)
        ax.set_xlim((0, 20))
        ax.set_ylim((-1, 21))
        plt.axis('off')
        
        new = pd.DataFrame({x:y for x,y in objects}, index=[1])
        plt.axis('off')
        if count == 4:
            plt.savefig('./causal_data/pendulum/test/a_' + name +'.png',dpi=96)
            test = test.append(new, ignore_index=True)
            count = 0
        else:
            plt.savefig('./causal_data/pendulum/train/a_' + name +'.png',dpi=96)
            train = train.append(new, ignore_index=True)
        # plt.show()
        plt.clf()
        count += 1
#%%
# train_imgs = os.listdir('./causal_data/pendulum/train')
# len(train_imgs)
# label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
# label.std(axis=0)
# label.mean(axis=0).round(2)
#%%