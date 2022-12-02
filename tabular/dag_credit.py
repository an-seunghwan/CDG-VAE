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
    load dataset: Home Credit Default Risk
    Reference: https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv
    """
    if not os.path.exists('./assets/credit'):
        os.makedirs('./assets/credit')
        
    base = pd.read_csv('./data/home-credit-default-risk/application_train.csv')
    base = base.sample(frac=1, random_state=1).reset_index(drop=True)
    base.head()
    #%%
    # https://chocoffee20.tistory.com/6
    covariates = [
        # 'TARGET', 
        'DAYS_EMPLOYED',
        'DAYS_BIRTH', 
        'AMT_ANNUITY', 
        'AMT_CREDIT', 
        'AMT_INCOME_TOTAL', 
        'AMT_GOODS_PRICE',
        # 'REGION_POPULATION_RELATIVE', 
        ]
    df = base[covariates]
    df = df.dropna(axis=0)
    
    # # imbalanced class
    # np.random.seed(1)
    # idx = np.random.choice(
    #     range((df['TARGET'] == 0).sum()), 
    #     (df['TARGET'] == 1).sum() * 1, 
    #     replace=False)
    # df = pd.concat([df.iloc[idx], df[df['TARGET'] == 1]], axis=0).reset_index().drop(columns=['index'])
    # df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    df = df.sample(frac=0.1, random_state=1).reset_index(drop=True)
    df.describe()
    #%%    
    """remove outlier"""
    for col in covariates:
        if col == 'TARGET': continue
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        condition = df[col] > q3 + 1.5 * iqr
        idx = df[condition].index
        df.drop(idx, inplace=True)
        condition = df[col] < q1 - 1.5 * iqr
        idx = df[condition].index
        df.drop(idx, inplace=True)
    print(df.shape)
    df.describe()
    #%%
    scaling = [x for x in covariates if x != 'TARGET']
    df_ = df.copy()
    df_[scaling] = (df[scaling] - df[scaling].mean(axis=0)) / df[scaling].std(axis=0)
    df_.describe()
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
    pdy.write_png('./assets/credit/dag_credit.png')
    fig = Image.open('./assets/credit/dag_credit.png')
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
        ['TARGET'], 
        ['AMT_ANNUITY'],
        ['AMT_CREDIT', 'DAYS_EMPLOYED', 'DAYS_BIRTH'],
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
    #%%
    cg = pc(data=bijection[:30000, :], 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=['u1', 'u2', 'u3'])
    pdy.write_png('./assets/credit/dag_bijection_credit.png')
    fig = Image.open('./assets/credit/dag_bijection_credit.png')
    fig.show()
    #%%
#%%
if __name__ == '__main__':
    main()
#%%