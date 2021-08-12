import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from joblib import Parallel, delayed
import xlsxwriter
from openpyxl import load_workbook
import os
import data_preprocess as pre

def runSKFold(n_seed, splits, data,target, columns_org, continuous, columns_dummified):
    '''
    Splitting the data into n_seed of splits folds cross validarion.
    
    Args:
        n_seed: number of cross-validation
        splits: number of folds in each cross validation
    
    Returns:
        data for each of n_seed * splits folds
    '''
    runs = []
    X = np.array(data.drop(target,axis=1))
    y = np.array(data[target])
    result = Parallel(n_jobs=-1)(delayed(execute_skfold)(X,y,columns_org, continuous, columns_dummified.drop(target), splits, seed) for seed in range(n_seed))
    for i in  result:
        for j in i:
            runs.append(j)
    return runs

def execute_skfold(X,y, columns_org, continuous, columns_dummified, splits, seed):
    '''
    Splitting the data into splits for each cross-validation.
    
    Args:
        X: The dataset
        y: The target values
        columns_org: column names before dummification
        continuous: column names of continuous values
        columns_dummified: column names after dummification
        splits: number of folds in each cross validation
        seed: The number of a single cross-validation
    
    Returns:
        data for each of the splits for a single cross-validation
    '''
    skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
    result = Parallel(n_jobs=-1)(delayed(execute_split)(X,y, columns_org, continuous, columns_dummified, train, test) for train, test in skf.split(X,y))
    return result

def execute_split(X,y, columns_org, continuous, columns_dummified, train, test):
    '''
    Impute and derive the class of all missing data from each split.
    
    Args:
        X: The dataset
        y: The target values
        columns_org: column names before dummification
        continuous: column names of continuous values
        columns_dummified: column names after dummification
        train: indices of the training set
        test: indices of the test set
    
    Returns:
        Imputed and fully processed data for each split
    '''
    X = X.copy()
    [X, MinMax] = pre.scale(pd.DataFrame(X, columns = columns_dummified), continuous)
    X = np.array(X)
    X_train, X_test = X[train], X[test] 

    X_train = pd.DataFrame(X_train, columns = columns_dummified)
    X_test = pd.DataFrame(X_test, columns = columns_dummified)

    imp_dum = KNNImputer(n_neighbors = 5)

    X_train.iloc[:] = imp_dum.fit_transform(X_train)
    X_test.iloc[:]  = imp_dum.transform(X_test)

    X_train = derive_class(X_train, columns_org.drop(continuous))
    X_test  = derive_class(X_test, columns_org.drop(continuous))

    X_train = np.array(pre.rescale(X_train, continuous, MinMax))
    X_test = np.array(pre.rescale(X_test, continuous, MinMax))
    y_train, y_test = y[train], y[test]

    arr = [X_train, X_test, y_train, y_test]
    return arr

def derive_class(data, columns_org):
    '''
    Uses a KNN imputed Data Frame with fractions, and uses probabilities and random choice generators to impute the missing values.

    Args:
        data: the data before deriving
        columns_org: column names before dummification
    
    Returns:
        Data after deriving the classes
    '''
    np.random.seed(5)
    for group in columns_org:
        indices = np.asarray(np.where(data.columns.str.startswith(group))).flatten().tolist()
        g_rows = []
        for col in data.columns[indices]:
            for rows in data.index[((data.loc[:,col]) <1) & (data.loc[:,col] >0)].tolist():
               g_rows.append(rows)
        g_rows = set(g_rows)
        for row in g_rows:
            prob=data.iloc[row,indices].tolist()
            if sum(prob) <1:
                indices.append(-1)
                prob.append(1-sum(prob))
            choice = np.random.choice(indices, p=prob)
            if -1 in indices: indices.remove(-1)
            data.iloc[row, indices] = 0
            if choice != -1: data.iloc[row, [choice]] = 1
    return data

def baseline(data, target, path):
    '''
    Generate baseline accuracy and f1 score and multivariate and univariate, and save them in a txt file.

    Args:
        data: The dataset after preprocessing
        target: The dependent variable of the dataset
        path: Directory from which the python file is being run
    '''
    f = open(path+'Baseline.txt','w+')
    rate = sum(data[target])/data.shape[0]
    rate2 = 1-rate
    f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
    f1 = 2*rate/(1+rate)
    f.write('base line f1 value is '+str(f1))
    f.close()