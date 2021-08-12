import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import  SGDClassifier, LogisticRegression


def elasticnet(X_train,X_test,y_train,y_test):
    '''
    Create multiple Elasticnet classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['Alpha','Confusion Matrix'])
    rows = []
    alphas= [0.0001,0.0005,  0.0008, 0.001,0.002,0.003,0.004,0.005, 0.01]
    for al in alphas:
        regr = SGDClassifier(loss = 'log',alpha= al,penalty = 'l1',random_state=0)
        model = regr.fit(X_train, y_train)
        predicted_labels = model.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([al,convert_matrix])
    for i in range(len(rows)):
        df = df.append({'Alpha':rows[i][0],'Confusion Matrix':rows[i][1]}, ignore_index=True)

    return df

def logistic_regression(X_train, X_test, y_train, y_test):
    '''
    Creates multiple Logitic Regression classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    model = LogisticRegression(penalty = 'none', max_iter=10000)
    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix = [tn,fp,fn,tp]
    df = pd.DataFrame()
    df['Confusion Matrix'] = [convert_matrix]
    return df

def KNN(X_train,X_test,y_train,y_test):
    '''
    Creates multiple KNearestNeighbors classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    neighbors = [5,10,12,14,16,20]
    df = pd.DataFrame(columns=['Neighbors','Confusion Matrix'])
    rows = []

    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        knn.fit(X_train,y_train)
        predicted_labels = knn.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([n, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Neighbors':rows[i][0],'Confusion Matrix':rows[i][1]}, ignore_index=True)

    return df


def SVM(X_train,X_test,y_train,y_test):
    '''
    Creates Support Vector Machine classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['Kernel','C','Gamma','Degree','Confusion Matrix'])
    rows = []

    Cs = [1e-1, 1, 1e1, 1e2, 1e3]
    gammas = [1,1e1]
    degrees = [2,3]

    for c in Cs:
        linear = LinearSVC(C=c, random_state=0, max_iter=100000, dual=False)
        linear.fit(X_train, y_train)
        predicted_labels = linear.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append(['linear', c, '', '', convert_matrix])

        for gamma in gammas:
            rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=100000)
            rbf.fit(X_train, y_train)
            predicted_labels = rbf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append(['rbf', c, gamma, '', convert_matrix])

            for degree in degrees:
                poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000000)
                poly.fit(X_train,y_train)
                predicted_labels = poly.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append(['poly', c, gamma, degree, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Kernel':rows[i][0],'C':rows[i][1],'Gamma':rows[i][2], 'Degree':rows[i][3],
        'Confusion Matrix':rows[i][4]}, ignore_index=True)

    return df



def rdforest(X_train,X_test,y_train,y_test):
    '''
    Creates multiple Random Forest classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['N_Estimators','Max_Depth','Confusion Matrix'])
    rows = []

    estimators = [200,300,400,500]
    max_depths = [5,7,10]

    for estimator in estimators:
        for max_d in max_depths:
            rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
            rdf.fit(X_train, y_train)
            predicted_labels = rdf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([estimator, max_d, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Depth':rows[i][1],'Confusion Matrix':rows[i][2]}, ignore_index=True)
    return df


def xgboost(X_train,X_test,y_train,y_test):
    '''
    Creates multiple XgBoost classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['Max_depth','N_estimators','Confusion Matrix'])
    rows = []

    rate = 0.05
    max_depth = [3,4,5,6,7]
    n_estimators= np.linspace(50, 450, 4, dtype=int)

    for depth in max_depth:
        for estimators in n_estimators:
            xgb = XGBClassifier(booster='gbtree',max_depth=depth,learning_rate=rate,n_estimators = estimators, use_label_encoder =False)
            xgb.fit(X_train, y_train)
            predicted_labels = xgb.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([depth,estimators,convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Max_depth':rows[i][0],'N_estimators':rows[i][1],
                        'Confusion Matrix':rows[i][2]}, ignore_index=True)
    return df


def naive_bayes(X_train,X_test,y_train,y_test):
    '''
    Creates multiple Naive Bayes classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.
    
    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(columns=['Confusion Matrix'])
    rows = []

    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    predicted_labels = bnb.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_b = [tn,fp,fn,tp]

    df = df.append({ 'Confusion Matrix':convert_matrix_b}, ignore_index=True)
    return df



def classify(estimator, X_train, X_test, y_train, y_test):
    '''
    Runs the specific Classification method.

    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model of the specified Classifier
    '''
    if estimator == 'svm':
        return SVM(X_train, X_test, y_train, y_test)
    elif estimator == 'naive_bayes':
        return naive_bayes(X_train, X_test, y_train, y_test)
    elif estimator == 'rdforest':
        return rdforest(X_train, X_test, y_train, y_test)
    elif estimator == 'knn':
        return KNN(X_train, X_test, y_train, y_test)
    elif estimator == 'elasticnet':
        return elasticnet(X_train, X_test, y_train, y_test)
    elif estimator =='xgboost':
        return xgboost(X_train, X_test, y_train, y_test)
    elif estimator =='logreg':
        return logistic_regression(X_train, X_test, y_train, y_test)