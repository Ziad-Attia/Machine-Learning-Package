import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF
import FCBF
import su_calculation as su
import MRMR as mr
import JMI


def infogain(X, y, n_features):
    '''
    Runs infogain feature selection on the data (X) and the target values (y) and finds
    the index of the top n_features number of features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    score = mutual_info_classif(X, y,random_state=0)
    index = list(np.argsort(list(score))[-1*n_features:])
    index.sort()
    return index

def reliefF(X, y, n_features):
    '''
    Runs ReliefF algorithm on the data (X) and the target values (y) and finds 
    the index of the top n_features number of features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    fs = ReliefF(n_neighbors=100, n_features_to_keep=n_features)
    fs.fit_transform(X, y)
    index = fs.top_features[:n_features]
    return index

def fcbf(X, y):
    '''
    Runs Fast Correlation-Based Filter feature selection on the data (X) and the target values (y) and finds
    the index of the significant features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    selection = FCBF.fcbf(X, y)
    index = list(selection[0])
    index.sort()
    return index

def merit_calculation(X, y):
    '''
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    
    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        
    Returns: Merits (float) of a feature subset X
    '''
    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su.su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su.su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def cfs(X, y):
    '''
    Runs Correlation-based feature selection on the data (X) and the target values (y) and finds
    the index of the significant features.

    Args:
        X: A Numpy array containing the dataset
        y: A Numpy array consisting of the target values
        n_features: An integer specifying how many features should be selected

    Returns a list containing the indices of the features that have been selected
    '''
    n_samples, n_features = X.shape
    F = []

    M = []
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 6:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break
    return np.array(F)
    

def run_feature_selection(method,X,y):
    '''
    Runs the specific ranking based feature selection method.

    Args:
        method: A string that refers to the feature selection method to be used
        X: An Arraylike structure containing the dataset
        y: An Arraylike structure consisting of the target values

    Returns:
        A list containing the indices of the features that have been selected
    '''
    X=np.array(X)
    y=np.array(y)
    if method[:3] == 'cfs':
        return cfs(X, y)
    elif method[:3] == 'jmi':
        if(len(method.split("_"))==1):
            return JMI.jmi(X,y)
        else:
            return JMI.jmi(X,y, n_selected_features = int(method.split("_")[1])) 
    elif method[:4] == 'mrmr':
        if(len(method.split("_"))==1):
            return mr.mrmr(X,y)
        else:
            return mr.mrmr(X,y, n_selected_features = int(method.split("_")[1]))    
    elif method[:4] =='fcbf':
        return fcbf(X,y)
    elif method[:7] == 'reliefF':
        return reliefF(X,y,int(method[8:]))
    elif method[:8] == 'infogain':
        return infogain(X,y,int(method[9:]))