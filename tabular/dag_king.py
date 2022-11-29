#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
#%%
def main():
    #%%
    """
    load dataset: House Sales in King County, USA
    Reference: 
    https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
    """
    df = pd.read_csv('./data/kc_house_data.csv')
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    # df['price'] = df['price'].map(lambda x: 1 if 450000 < x else 0)
    df.head()
    #%%
    continuous = [
        'price',
        'bedrooms', 'bathrooms', 'floors', 
        'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 
        # 'yr_built', 
        ]
    df = df[continuous]
    # categorical = ['bedrooms', 'bathrooms', 'floors', 
    #                'condition', 'grade', 'waterfront', 'view', 
    #                'price']
    # df = df[categorical]
    df_ = (df - df.mean(axis=0)) / df.std(axis=0)
    
    if not os.path.exists('./assets/king'):
        os.makedirs('./assets/king')
    #%%
    """PC algorithm : CPDAG"""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    cg = pc(data=df_.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/king/dag_king.png')
    fig = Image.open('./assets/king/dag_king.png')
    fig.show()
    #%%
    """bijection"""
    def interleave_float(a: float, b: float):
        a_rest = a
        b_rest = b
        result = 0
        dst_pos = 1.0  # position of written digit
        while a_rest != 0 or b_rest != 0:
            dst_pos /= 10  # move decimal point of write
            a_rest *= 10  # move decimal point of read
            result += dst_pos * (a_rest // 1)
            a_rest %= 1  # remove current digit

            dst_pos /= 10
            b_rest *= 10
            result += dst_pos * (b_rest // 1)
            b_rest %= 1
        return result
    
    # df.describe()
    # df['bedrooms'] = df['bedrooms'] * 0.01
    # df['bathrooms'] = df['bathrooms'] * 0.1
    # df['floors'] = df['floors'] * 0.1
    # df['condition'] = df['condition'] * 0.1
    # df['grade'] = df['grade'] * 0.01
    # df['waterfront'] = df['waterfront'] * 0.1
    # df['view'] = df['view'] * 0.1
    # df['price'] = df['price'] * 0.1
    
    min_ = df_.min(axis=0)
    max_ = df_.max(axis=0)
    df = (df_ - min_) / (max_ - min_) 
    #%%
    topology = [
        ['bedrooms'], ['bathrooms'], ['floors'], 
        ['sqft_basement', 'sqft_living', 'sqft_above'],
        ['price', 'sqft_lot'],
        ]
    bijection = []
    i = -1
    for i in range(len(topology)):
        if len(topology[i]) == 1:
            bijection.append(df[topology[i]].to_numpy())
            continue
        if len(topology[i]) == 2:
            df_tmp = df[topology[i]].to_numpy()
            bijection_tmp = []
            for x, y in df_tmp:
                bijection_tmp.append(interleave_float(x, y))
            bijection.append(np.array([bijection_tmp]).T)
        if len(topology[i]) == 3:
            df_tmp = df[topology[i]].to_numpy()
            bijection_tmp = []
            for x, y in df_tmp[:, :2]:
                bijection_tmp.append(interleave_float(x, y))
            tmp = np.concatenate([np.array([bijection_tmp]).T, df_tmp[:, [2]]], axis=1)
            bijection_tmp = []
            for x, y in tmp:
                bijection_tmp.append(interleave_float(x, y))
            bijection.append(np.array([bijection_tmp]).T)
    bijection = np.concatenate(bijection, axis=1)
    
    # topology = [[['bedrooms', 'bathrooms'], ['waterfront', 'view']], 
    #             [['grade', 'price'], ['condition', 'floors']]]
    # i = 0
    # bijection_result = []
    # for i in range(len(topology)):
    #     # bijection = []
    #     for top in topology[i]:
    #         df_tmp = df[top].to_numpy()
    #         bijection_tmp = []
    #         for x, y in df_tmp:
    #             bijection_tmp.append(interleave_float(x, y))
    #         bijection_result.append(np.array([bijection_tmp]).T)
    #     # bijection = np.concatenate(bijection, axis=1)
        
    #     # bijection_tmp = []
    #     # for x, y in bijection:
    #     #     bijection_tmp.append(interleave_float(x, y))
    #     # bijection_result.append(np.array([bijection_tmp]).T)
        
    # bijection_result = np.concatenate(bijection_result, axis=1)
    #%%
    cg = pc(data=bijection, 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G)
    pdy.write_png('./assets/king/dag_bijection_king.png')
    fig = Image.open('./assets/king/dag_bijection_king.png')
    fig.show()
    #%%
#%%
if __name__ == '__main__':
    main()
#%%