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
    
    train = df.iloc[:4000]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    train = (train - mean) / std # local statistic
    #%%
    """PC algorithm : CPDAG"""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    cg = pc(data=train.to_numpy(), 
            alpha=0.1, 
            indep_test='fisherz') 
    print(cg.G)
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=train.columns)
    pdy.write_png('./assets/dag_loan.png')
    fig = Image.open('./assets/dag_loan.png')
    fig.show()
    #%%
    """
    US Census data
    Reference:
    [1] https://arxiv.org/pdf/2108.04884v3.pdf
    [2] https://github.com/zykls/folktables
    """
    import folktables
    from folktables import ACSDataSource, ACSIncome
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    df = data_source.get_data(states=["CA"], download=True)
    ACSIncome = folktables.BasicProblem(
        features=[
            # 'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            # 'WKHP',
            'SEX',
            'RAC1P',
        ],
        target='PINCP',
        target_transform=lambda x: x > 50000,    
        group='RAC1P',
        preprocess=folktables.adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    data, label, group = ACSIncome.df_to_numpy(df)
    df = pd.DataFrame(data, columns=ACSIncome.features)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    #%%
    """PC algorithm : CPDAG"""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    alpha = 0.01
    
    cg = pc(data=df.to_numpy()[:5000, :], 
            alpha=alpha, 
            indep_test='chisq') 
    print(cg.G)
    #%%
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/dag_acsincome.png')
    fig = Image.open('./assets/dag_acsincome.png')
    fig.show()
    #%%
    #%%
#%%
if __name__ == '__main__':
    main()
#%%