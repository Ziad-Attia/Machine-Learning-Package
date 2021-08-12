import pandas as pd
import os
import runs

def main():
    datafile="" #name of the data file in the data folder
    target = "" #name of the dependent variable

    #Keep the classification and feature selection methods that you want
    classifiers=['logreg', 'xgboost', 'elasticnet', 'knn', 'rdforest', 'naive_bayes', 'svm']
    fselect=['AllFeatures', 'infogain_#', 'reliefF_#', 'jmi_#', 'mrmr_#', 'cfs', 'fcbf'] # replace the # with the number of features you want
    #Note that cfs and fcbf find all the significant features so they don't need a number


    n_seed = 10 #number of validations
    splits = 10 #number of folds or splits in each validation run

    directory_path = os.path.dirname(os.path.realpath(__file__))+'/'


    #Specify which data file type youa are using
    data = pd.read_spss(directory_path+"data/"+datafile+".sav")
    #data = pd.read_csv(directory_path+datafile+".csv") 

    #You can either choose to do a SMOTE run for imbalanced data or a Normal run
    runs.NormalRun(data,directory_path, datafile, target, classifiers, fselect, n_seed, splits)
    #runs.SmoteAnalysis(data,directory_path, datafile, target, classifiers, fselect, n_seed, splits)

main()