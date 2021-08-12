import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import pyreadstat
import data_preprocess as pre
import uni_multiStats as stats 
import scoring as score 
import ranking_subset_run as rsr
import boost_bag_run as bbg
import stats as st
import FeatureOrganize as fo

def NormalRun(data, directory_path, datafile, target, classifiers, fselect, n_seed, splits):
    '''
    Create all the directories to store all the intermediate and final results. 
    Impute and prepare the data for running.
    Splits the data and run the feature selection and classification on runs.

    Args:
        data : The DataFrame that contains data that hasn't been preprocessed.
        directory_path: Directory from which the python file is being run
        datafile: name of the datafile inside the data folder
        target: the dependent variable of the dataset
        classifiers: Classification methods used
        fselect: Feature selection methods used
        type: Type of analysis performed
        n_seed: Number of validations
        splits: Number of folds or splits in each validation run   
    '''
    
    target_path = directory_path+'NormalDataAnalysis/'+datafile+"_"+target+"/"

    feature_path = target_path+"features/"
    STATS_path = target_path+"STATS/"
    results_path = target_path+"resultsparallel/"
    data_path = target_path+"dataparallel/"

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    if not os.path.exists(STATS_path):
        os.makedirs(STATS_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    datacopy = data.copy(deep=True)
    datacopy.replace('', np.nan,regex=True, inplace=True)
    datacopy = pre.remove_invalid_data(datacopy, target)

    stats.baseline(datacopy, target, STATS_path)

    columns_org = datacopy.columns
    [datacopy, continuous] = pre.modify_data(datacopy)
    columns_dummified = datacopy.columns
    n_features = datacopy.shape[1]-1

    runs = stats.runSKFold(n_seed, splits, data=datacopy,target=target, columns_org=columns_org, continuous=continuous, columns_dummified=columns_dummified)

    for c in classifiers:
        for fs in fselect:
            score.score(rsr.normal_run(target_path, columns_dummified.drop([target]), n_seed, splits, fs, c, runs, n_features),n_seed)

    st.create_STATS(target_path)
    st.heatmap(target_path, target)



def SmoteAnalysis(data,directory_path, datafile, target, classifiers, fselect, n_seed, splits):
    '''
    Create all the directories to store all the intermediate and final results. 
    Impute, oversample, and prepare the data for running.
    Splits the data and run the feature selection and classification on runs.

    Args:
        data : The DataFrame that contains data that hasn't been preprocessed.
        directory_path: Directory from which the python file is being run
        datafile: name of the datafile inside the data folder
        target: the dependent variable of the dataset
        classifiers: Classification methods used
        fselect: Feature selection methods used
        type: Type of analysis performed
        n_seed: Number of validations
        splits: Number of folds or splits in each validation run      
    '''

    target_path = directory_path+'SmoteDataAnalysis/'+datafile+"_"+target+"/"

    feature_path = target_path+"features/"
    STATS_path = target_path+"STATS/"
    results_path = target_path+"resultsparallel/"
    data_path = target_path+"dataparallel/"

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    if not os.path.exists(STATS_path):
        os.makedirs(STATS_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    datacopy = data.copy(deep=True)
    datacopy.replace('', np.nan,regex=True, inplace=True)
    datacopy = pre.remove_invalid_data(datacopy, target)

    stats.baseline(datacopy, target, STATS_path)

    columns_org = datacopy.columns
    [datacopy, continuous] = pre.modify_data(datacopy)
    n_features = datacopy.shape[1]-1
    [datacopy, MinMax] = pre.scale(datacopy, continuous)

    y = datacopy[target]
    X = datacopy.drop(columns=[target])
    
    imp_dum = KNNImputer(n_neighbors = 5)
    X.iloc[:] = imp_dum.fit_transform(X)

    X = stats.derive_class(X, columns_org.drop(continuous))

    over = SMOTE(sampling_strategy=1, random_state=5)
    X_res, y_res = over.fit_resample(X, y)

    data_SMOTE = pd.concat([X_res, y_res], axis=1)
    data_SMOTE = pre.rescale(data_SMOTE, continuous, MinMax)
    
    columns_dummified = data_SMOTE.columns
    runs = stats.runSKFold(n_seed,splits,data=data_SMOTE,target=target, columns_org=columns_org, continuous=continuous, columns_dummified=columns_dummified)

    for c in classifiers:
        for fs in fselect:
            score.score(rsr.normal_run(target_path, columns_dummified.drop([target]), n_seed, splits, fs, c, runs, n_features),n_seed)
                

    st.create_STATS(target_path)
    st.heatmap(target_path, target)