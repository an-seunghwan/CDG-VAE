#%%
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
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
varphi = -0.1 # -1 ~ 1
theta = 0.1 # -1 ~ 1
center = (10, 10.5)
threshold = 45
l = 8
for varphi in tqdm.tqdm(np.linspace(-1, 1, 100)):
    for theta in np.linspace(-1, 1, 100):
        objects = []
        
        light = center[0] - center[1] * math.tan(math.radians(varphi * threshold))
        ball = (center[0] + l * math.sin(math.radians(theta * threshold)),
                center[1] - l * math.cos(math.radians(theta * threshold)))

        plt.rcParams['figure.figsize'] = (1.0, 1.0)
                
        gun = plt.Polygon(([10, 10.5], ball), color = 'black', linewidth = 3)
        pendulum = plt.Circle(ball, 1.5, color = 'firebrick')
        sun = plt.Circle((light, 20.5), 3, color = 'orange')

        ball_x = center[0] + (l + 1.5) * math.sin(math.radians(theta * threshold))
        ball_y = center[0] - (l + 1.5) * math.cos(math.radians(theta * threshold))

        tan_phi = light - ball[0]
        tan_phi /= 20.5 - ball[1]
        shadow_start = center[0] + center[1] * math.tan(math.radians(varphi * threshold))
        length = light - 20.5 * tan_phi - shadow_start

        shadow = plt.Polygon(([shadow_start, -0.5], [shadow_start + length, -0.5]), color = 'black', linewidth = 4)
        position = shadow_start + length / 2
        
        objects.append(('light', light))
        objects.append(('theta', theta))
        objects.append(('length', length))
        objects.append(('position', position))
        name = '_'.join([str(int(y)) for x,y in objects])
        
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