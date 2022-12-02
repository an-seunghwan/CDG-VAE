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
    load dataset: Forest Cover Type Prediction
    Reference: https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv
    """
    if not os.path.exists('./assets/covtype'):
        os.makedirs('./assets/covtype')
        
    base = pd.read_csv('./data/covtype.csv')
    base = base.sample(frac=1, random_state=5).reset_index(drop=True)
    print(base.shape)
    base.head()
    #%%
    covariates = [
        'Horizontal_Distance_To_Hydrology', 
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
        'Elevation', 
        'Aspect', 
        'Slope', 
        'Cover_Type',
        ]
    df = base[covariates]
    df = df.dropna(axis=0)
    print(df.shape)
    df.describe()
    #%%    
    scaling = [x for x in covariates if x != 'Cover_Type']
    df_ = df.copy()
    df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
    df_.describe()
    #%%
    """PC algorithm : CPDAG"""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    cg = pc(data=df_.to_numpy()[2000:, :], 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/covtype/dag_covtype.png')
    fig = Image.open('./assets/covtype/dag_covtype.png')
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
    
    min_ = df_.min(axis=0)
    max_ = df_.max(axis=0)
    df = (df_ - min_) / (max_ - min_) 
    #%%
    topology = [
        ['Horizontal_Distance_To_Hydrology'], 
        ['Vertical_Distance_To_Hydrology'],
        ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points'],
        ['Elevation'], 
        ['Aspect'], 
        ['Slope', 'Cover_Type'],
        ]
    bijection = []
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
    bijection = np.concatenate(bijection, axis=1)
    #%%
    cg = pc(data=bijection[2000:, :], 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    # labels = ['u{}'.format(i+1) if len(topology[i]) != 1 else topology[i][0] for i in range(len(topology))]
    pdy = GraphUtils.to_pydot(cg.G, labels=['u1', 'u5', 'u6', 'u4', 'u2', 'u3'])
    pdy.write_png('./assets/covtype/dag_bijection_covtype.png')
    fig = Image.open('./assets/covtype/dag_bijection_covtype.png')
    fig.show()
    #%%
#%%
if __name__ == '__main__':
    main()
#%%