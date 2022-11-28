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
    df['price'] = df['price'].map(lambda x: 1 if 450000 < x else 0)
    df.head()
    #%%
    categorical = ['bedrooms', 'bathrooms', 'floors', 
                   'condition', 'grade', 'waterfront', 'view', 
                   'price',]
    df = df[categorical]
    
    if not os.path.exists('./assets/king'):
        os.makedirs('./assets/king')
    #%%
    """PC algorithm : CPDAG"""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    cg = pc(data=df.to_numpy(), 
            alpha=0.05, 
            indep_test='chisq') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/king/dag_king.png')
    fig = Image.open('./assets/king/dag_king.png')
    fig.show()
    #%%
    # """bijection"""
    # def interleave_float(a: float, b: float):
    #     a_rest = a
    #     b_rest = b
    #     result = 0
    #     dst_pos = 1.0  # position of written digit
    #     while a_rest != 0 or b_rest != 0:
    #         dst_pos /= 10  # move decimal point of write
    #         a_rest *= 10  # move decimal point of read
    #         result += dst_pos * (a_rest // 1)
    #         a_rest %= 1  # remove current digit

    #         dst_pos /= 10
    #         b_rest *= 10
    #         result += dst_pos * (b_rest // 1)
    #         b_rest %= 1
    #     return result
    
    # min_ = df_.min(axis=0)
    # max_ = df_.max(axis=0)
    # df = (df_ - min_) / (max_ - min_) 
    
    # topology = [['Mortgage', 'Income'], ['Experience', 'Age'], ['CCAvg']]
    # bijection = []
    # for i in range(len(topology)):
    #     if len(topology[i]) == 1:
    #         bijection.append(df[topology[i]].to_numpy())
    #         continue
    #     df_tmp = df[topology[i]].to_numpy()
    #     bijection_tmp = []
    #     for x, y in df_tmp:
    #         bijection_tmp.append(interleave_float(x, y))
    #     bijection.append(np.array([bijection_tmp]).T)
    # bijection = np.concatenate(bijection, axis=1)
    # #%%
    # cg = pc(data=bijection[:4000, :], 
    #         alpha=0.05, 
    #         indep_test='chisq') 
    # print(cg.G)
    
    # # visualization
    # pdy = GraphUtils.to_pydot(cg.G, labels=['u1', 'u2', 'u3'])
    # pdy.write_png('./assets/loan/dag_bijection_loan.png')
    # fig = Image.open('./assets/loan/dag_bijection_loan.png')
    # fig.show()
    #%%
#%%
if __name__ == '__main__':
    main()
#%%