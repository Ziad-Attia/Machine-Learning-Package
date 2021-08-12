import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def remove_invalid_data(data,target):
    """
    Function that removes columns that aren't suitable for machine learning.
    This includes features with more than 10% missing values, wrong data type,
    and the indices.

    Args:
        data: DataFrame that contains data that hasn't been preprocessed.
        target: The dependent variable of the dataset

    Returns:
        DataFrame: Preprocessed DataFrame
    """    
    
    data = data[data[target].notna()]
    data = data[data.columns[data.isnull().mean()<0.1]]
    data = data.select_dtypes(exclude=['object'])
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis = 1)
    return data


def sep_nondummies(data):
    """
    Finds the features that are:
        1. have nominal values
        2. have more than 2 distinct values so it needs to be dummified

    Args:
        data: DataFrame containing the dataset

    Returns:
        nominal: Array-like structure that contains the nominal features
        continuous: Array-like structure that contains the names of all the continuous features
    """
    nominal = []
    continuous=[]
    for col in data.columns:
        distinct = data[col].dropna().nunique()
        if distinct > 10:
            continuous.append(col)  
        elif distinct > 2:  
            nominal.append(col) 
    return [nominal, continuous]

def create_dummies (data):
    """
    Creates dummy variables.

    Args:
        data (DataFrame): DataFrame containing the dataset

    Returns:
        DataFrame: DataFrame containing the dataset with dummy variables
    """

    dummy = pd.get_dummies(data, columns = data.columns, drop_first= True) 
    return dummy


            

def modify_data(data):
    """
    Runs all the preprocessing functions on the dataset.

    Args:
        data (DataFrame): DataFrame containing the dataset with no preprocessing
        target: The dependent variable  of the dataset

    Returns:
        DataFrame: DataFrame with all the preprocessing done
        continuous: Array-like structure that contains the names of all the continuous features
    """
    [nominal, continuous] = sep_nondummies(data) 
    if nominal:
        nominal_dummified = create_dummies(data[nominal])
        for column in nominal_dummified.columns:
            nominal_dummified.loc[nominal_dummified[column].isnull(),nominal_dummified.columns.str.startswith(column+'_')] = np.nan
        data = data.drop(nominal, axis = 1)
        data = pd.concat([data,nominal_dummified], axis =1)
    return [data, continuous]

def scale(data, continuous):
    '''
    Normalize the continuous variables and stores the minimum and maximum values of each variable if we want to rescale them back.
    
    Args:
        data: DataFrame containing the dataset
        continuous: Array-like structure that contains the names of all the continuous features

    Returns:
        data: DataFrame after the Normalization
        MinMax: A DataFrame that contains the minimum and maximum values of each variable
    '''
    if continuous: 
        MinMax = pd.DataFrame(columns = continuous, index = ['min', 'max'])
        for col in continuous:
            MinMax.loc['max', col] = data[col].max()
            MinMax.loc['min', col] = data[col].min()
        scaler = MinMaxScaler()
        data[continuous] = scaler.fit_transform(data[continuous])
        return [data, MinMax]
    return [data, None]

def rescale(data, continuous, MinMax):
    '''
    Reverse the Normalization done in the scale function above.

    Args:
        data: DataFrame containing the dataset
        continuous: Array-like structure that contains the names of all the continuous features
        MinMax: A DataFrame that contains the minimum and maximum values of each variable

    Returns:
        data: DataFrame after reversing Normalization
    '''
    for col in continuous:
        data[col] = (data[col]*(MinMax.loc['max', col] - MinMax.loc['min', col])) + MinMax.loc['min', col]
    return data