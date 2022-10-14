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

from PIL import Image
import math
import numpy as np
import pandas as pd 
import tqdm
import warnings
warnings.filterwarnings('ignore')
#%%
"""
For main experiments and sample efficiency
Add followings:
[1]: measurement error
[2]: environmental disturbance (corruption)
[3]: target label
"""
#%%
foldername = 'pendulum_real'
if not os.path.exists('./causal_data/{}/'.format(foldername)): 
    os.makedirs('./causal_data/{}/train'.format(foldername))
    os.makedirs('./causal_data/{}/test'.format(foldername))
#%%
train = pd.DataFrame(columns=['light', 'angle', 'length', 'position'])
test = pd.DataFrame(columns=['light', 'angle', 'length', 'position'])
#%%
"""Data Generating Process"""
np.random.seed(1)

light_angle_list= np.random.uniform(math.pi/4, math.pi/2, 10000)
pendulum_angle_list = np.random.uniform(0, math.pi/4, 10000)
# light_angle_list= np.linspace(math.pi/4, math.pi/2, 10)
# pendulum_angle_list = np.linspace(0, math.pi/4, 10)

center = (10, 10.5) # (c_x, c_y) : the axis's of the center     
l = 9.5  # length of pendulum (including the red ball)
b = -0.5
#%%
count = 0
scale = 0.1 # measurement error scale
beta = [1, -1, 0.5, -0.5]
for light_angle, pendulum_angle in tqdm.tqdm(zip(light_angle_list, pendulum_angle_list)):
    objects = []
    
    xi_1 = light_angle
    xi_2 = pendulum_angle
    
    light = center[0] + (10 / math.tan(xi_1))
    
    x = 10 + (l - 1.5) * math.sin(xi_2)
    y = 10 - (l - 1.5) * math.cos(xi_2)
    
    # xi_3 : shadow_length
    # xi_4 : shadow_position
    xi_3 = (center[0] + l * math.sin(xi_2) - (center[1] - l * math.cos(xi_2) - b) / math.tan(xi_1) ) 
    xi_3 -= (center[0] - (center[1] - b) / math.tan(xi_1))
    xi_4 = center[0] + l * math.sin(xi_2) - (center[1] - l * math.cos(xi_2) - b) / math.tan(xi_1) 
    xi_4 += (center[0] - (center[1] - b) / math.tan(xi_1))
    xi_4 /= 2
    
    """measurement error"""
    xi_3 += np.random.normal(loc=0, scale=scale)
    xi_4 += np.random.normal(loc=0, scale=scale)
    
    """data corruption: 20%"""
    if (count + 1) % 5 == 0:
        xi_3 = np.random.uniform(low=0, high=12)
        xi_3 = np.random.uniform(low=0, high=12)
    
    """target label"""
    logit = sum([b*x for b,x in zip(beta, [xi_1, xi_2, xi_3, xi_4])])
    tau = np.random.binomial(n=1, p=1 / (1 + np.exp(-logit + 2 * np.sin(logit)))) # nonlinear but causal
    
    objects.append(('light', xi_1))
    objects.append(('angle', xi_2))
    objects.append(('length', xi_3))
    objects.append(('position', xi_4))
    objects.append(('target', tau))
    name = '_'.join([str(round(j, 4)) for i,j in objects])
    
    plt.rcParams['figure.figsize'] = (1.0, 1.0)
    
    sun = plt.Circle((light, 20.5), 3, color = 'orange') 
    gun = plt.Polygon(([10, 10.5], [x,y]), color = 'black', linewidth = 3)
    ball = plt.Circle((x, y), 1.5, color = 'firebrick')
    shadow = plt.Polygon(([xi_4 - xi_3 / 2.0, -0.5], [xi_4 + xi_3 / 2.0, -0.5]), color = 'black', linewidth = 3)
    
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
    if (count + 1) % 4 == 0:
        plt.savefig('./causal_data/{}/test/a_' .format(foldername)+ name +'.png',dpi=96)
        test = test.append(new, ignore_index=True)
    else:
        plt.savefig('./causal_data/{}/train/a_'.format(foldername) + name +'.png',dpi=96)
        train = train.append(new, ignore_index=True)
    # plt.show()
    plt.clf()
    
    count += 1
#%%
# train_imgs = [x for x in os.listdir('./causal_data/{}/train'.format(foldername)) if x.endswith('.png')]
# len(train_imgs)
# test_imgs = [x for x in os.listdir('./causal_data/{}/test'.format(foldername)) if x.endswith('.png')]
# len(test_imgs)
# label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
# label.std(axis=0).round(2)
# label.mean(axis=0).round(2)
#%%
# train_imgs = [x for x in os.listdir('./causal_data/pendulum/train') if x.endswith('.png')]
# img = np.array(Image.open('./causal_data/pendulum/train/' + train_imgs[0]).resize((64, 64)))[:, :, :3].astype(float)
# img /= 255
# # light
# mask = np.zeros((64, 64, 3))
# mask[:20, :] = 1
# plt.imshow(img * mask)
# # angle
# mask = np.zeros((64, 64, 3))
# mask[20:51, :] = 1
# plt.imshow(img * mask)
# # shadow
# mask = np.zeros((64, 64, 3))
# mask[51:, :] = 1
# plt.imshow(img * mask)
#%%