#%%
import numpy as np
#%%
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.metrics import f1_score
#%%
def regression_eval(train, test, target):
    covariates = [x for x in train.columns if x not in [target]]
    
    result = []
    for name, regr in [
        ('linear', LinearRegression()), 
        ('RF', RandomForestRegressor(random_state=0)), 
        ('GradBoost', GradientBoostingRegressor(random_state=0))]:
        
        """baseline"""
        regr.fit(train[covariates], train[target])
        pred = regr.predict(test[covariates])
        
        rsq = (test[target] - pred).pow(2).sum()
        rsq /= np.var(test[target]) * len(test)
        rsq = 1 - rsq
        
        result.append((name, rsq))
        print("[{}] R^2: {:.3f}".format(name, rsq))
    return result
#%%
def classification_eval(train, test, target):
    covariates = [x for x in train.columns if not x.startswith(target)]
    
    result = []
    for name, clf in [
        ('logistic', LogisticRegression(multi_class='ovr')), 
        ('RF', RandomForestClassifier(random_state=0)), 
        ('GradBoost', GradientBoostingClassifier(random_state=0))]:
        
        """baseline"""
        clf.fit(train[covariates], train[target])
        pred = clf.predict(test[covariates])
        
        f1 = f1_score(test[target], pred, average='micro')
        
        result.append((name, f1))
        print("[{}] F1: {:.3f}".format(name, f1))
    return result
#%%