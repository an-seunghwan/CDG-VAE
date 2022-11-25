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
    load dataset: Personal Loan
    Reference: 
    https://www.kaggle.com/datasets/teertha/personal-loan-modeling
    """
    df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df = df.drop(columns=['ID'])
    continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
    df = df[continuous]
    
    if not os.path.exists('./assets/loan'):
        os.makedirs('./assets/loan')
    #%%
    """PC algorithm : CPDAG"""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    df_ = (df - df.mean(axis=0)) / df.std(axis=0)
    
    cg = pc(data=df_.to_numpy(), 
            alpha=0.05, 
            indep_test='chisq') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df_.columns)
    pdy.write_png('./assets/loan/dag_loan.png')
    fig = Image.open('./assets/loan/dag_loan.png')
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
    
    topology = [['Mortgage', 'Income'], ['Experience', 'Age'], ['CCAvg']]
    bijection = []
    for i in range(len(topology)):
        if len(topology[i]) == 1:
            bijection.append(df[topology[i]].to_numpy())
            continue
        df_tmp = df[topology[i]].to_numpy()
        bijection_tmp = []
        for x, y in df_tmp:
            bijection_tmp.append(interleave_float(x, y))
        bijection.append(np.array([bijection_tmp]).T)
    bijection = np.concatenate(bijection, axis=1)
    #%%
    cg = pc(data=bijection, 
            alpha=0.05, 
            indep_test='chisq') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=['u1', 'u2', 'u3'])
    pdy.write_png('./assets/loan/dag_bijection_loan.png')
    fig = Image.open('./assets/loan/dag_bijection_loan.png')
    fig.show()
    #%%
    # """
    # US Census data
    # Reference:
    # [1] https://arxiv.org/pdf/2108.04884v3.pdf
    # [2] https://github.com/zykls/folktables
    # """
    # import folktables
    # from folktables import ACSDataSource, ACSIncome
    # data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    # df = data_source.get_data(states=["CA"], download=True)
    # ACSIncome = folktables.BasicProblem(
    #     features=[
    #         # 'AGEP',
    #         'COW',
    #         'SCHL',
    #         'MAR',
    #         'OCCP',
    #         'POBP',
    #         'RELP',
    #         # 'WKHP',
    #         'SEX',
    #         'RAC1P',
    #     ],
    #     target='PINCP',
    #     target_transform=lambda x: x > 50000,    
    #     group='RAC1P',
    #     preprocess=folktables.adult_filter,
    #     postprocess=lambda x: np.nan_to_num(x, -1),
    # )
    # data, label, group = ACSIncome.df_to_numpy(df)
    # df = pd.DataFrame(data, columns=ACSIncome.features)
    # df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    # #%%
    # """PC algorithm : CPDAG"""
    # from causallearn.search.ConstraintBased.PC import pc
    # from causallearn.utils.GraphUtils import GraphUtils
    
    # alpha = 0.01
    
    # cg = pc(data=df.to_numpy()[:5000, :], 
    #         alpha=alpha, 
    #         indep_test='chisq') 
    # print(cg.G)
    # #%%
    # # visualization
    # pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    # pdy.write_png('./assets/dag_acsincome.png')
    # fig = Image.open('./assets/dag_acsincome.png')
    # fig.show()
    #%%
#%%
if __name__ == '__main__':
    main()
#%%