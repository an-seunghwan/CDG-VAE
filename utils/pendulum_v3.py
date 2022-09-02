#%%
"""
Reference
[1]: https://github.com/huawei-noah/trustworthyAI/blob/master/research/CausalVAE/causal_data/pendulum.py
[2]: https://arxiv.org/abs/2010.02637
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
    os.makedirs('./causal_data/pendulum/train')
    os.makedirs('./causal_data/pendulum/test')
#%%
train = pd.DataFrame(columns=['light', 'angle', 'length', 'position'])
test = pd.DataFrame(columns=['light', 'angle', 'length', 'position'])
#%%
"""Data Generating Process"""
np.random.seed(1)

# light_angle_list= np.random.uniform(math.pi/4, math.pi/2, 100)
# pendulum_angle_list = np.random.uniform(0, math.pi/4, 100)
light_angle_list= np.linspace(math.pi/4, math.pi/2, 100)
pendulum_angle_list = np.linspace(0, math.pi/4, 100)
center = (10, 10.5) # (c_x, c_y) : the axis's of the center     
l = 9.5  # length of pendulum (including the red ball)
b = -0.5
#%%
count = 0
for pendulum_angle in tqdm.tqdm(pendulum_angle_list):
    for light_angle in light_angle_list:
        objects = []
        
        xi_1 = pendulum_angle
        xi_2 = light_angle
        
        light = center[0] + (10 / math.tan(xi_2))
        
        x = 10 + (l - 1.5) * math.sin(xi_1)
        y = 10 - (l - 1.5) * math.cos(xi_1)
        
        # xi_3 : shadow_length
        # xi_4 : shadow_position
        xi_3 = (center[0] + l * math.sin(xi_1) - (center[1] - l * math.cos(xi_1) - b) / math.tan(xi_2) ) 
        xi_3 -= (center[0] - (center[1] - b) / math.tan(xi_2))
        xi_4 = center[0] + l * math.sin(xi_1) - (center[1] - l * math.cos(xi_1) - b) / math.tan(xi_2) 
        xi_4 += (center[0] - (center[1] - b) / math.tan(xi_2))
        xi_4 /= 2
        
        objects.append(('light', xi_2))
        objects.append(('angle', xi_1))
        objects.append(('length', xi_3))
        objects.append(('position', xi_4))
        name = '_'.join([str(round(j, 4)) for i,j in objects])
        
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
        
        sun = plt.Circle((light, 20.5), 3, color = 'orange') 
        gun = plt.Polygon(([10, 10.5],[x,y]), color = 'black', linewidth = 3)
        ball = plt.Circle((x,y), 1.5, color = 'firebrick')
        shadow = plt.Polygon(([xi_4 - xi_3 / 2.0, -0.5],[xi_4 + xi_3 / 2.0, -0.5]), color = 'black', linewidth = 3)
        
        ax = plt.gca()
        ax.add_artist(sun)
        ax.add_artist(gun)
        ax.add_artist(ball)
        ax.add_artist(shadow)
       
        ax.set_xlim((0, 20))
        ax.set_ylim((-2, 22))
        plt.axis('off')
        
        new = pd.DataFrame({i:j for i,j in objects}, index=[1])
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
# train_imgs = [x for x in os.listdir('./causal_data/pendulum/train') if x.endswith('.png')]
# len(train_imgs)
# test_imgs = [x for x in os.listdir('./causal_data/pendulum/test') if x.endswith('.png')]
# len(test_imgs)
# label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
# label.std(axis=0).round(2)
# label.mean(axis=0).round(2)
#%%