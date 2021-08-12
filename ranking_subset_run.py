import numpy as np
import pandas as pd
import featureselection as fselect
import classifiers as e
import os
from joblib import Parallel, delayed
import pickle

def execute_feature_selection(path, runs, method, n_features):
    result = Parallel(n_jobs=-1)(delayed(execute_feature_selection_a_run)(run, method, n_features) for  run in runs)
    with open(path+"features/"+method+".txt","wb") as  fp:
        pickle.dump(result,fp)
    return result

def execute_feature_selection_a_run(run, method, n_features):
    X_train,  y_train = run[0], run[2]
    arr = []
    if(method == 'AllFeatures'):
        arr.append(range(n_features))
    else:
        arr.append(fselect.run_feature_selection(method, X_train, y_train))
    return arr

def execute_a_run(path, index, run,features,estimator, method):
    X_train, X_test = run[0], run[1]
    y_train, y_test = run[2], run[3]
    X_train, X_test = X_train[:,features[0]], X_test[:,features[0]]

    result = e.classify(estimator, X_train, X_test, y_train, y_test)
    result.rename(columns={"Confusion Matrix": "Confusion Matrix Split" + " " + str(index+1)}, inplace=True)
    filename = path+"dataparallel/"+estimator+method+'_'+str(index)
    result.to_csv(filename+'.csv')
            

def combinefile(path, method,estimator,splits, n_seed,outer_dict):
    for seed in range(n_seed):
        df = pd.DataFrame()
        temp = pd.read_csv(str(path+"dataparallel/"+estimator+method+'_'+str((seed)*splits)+'.csv'), index_col=0)
        df = pd.concat([df,temp],axis=1)
        for i  in range(1,splits):
            temp = pd.read_csv(str(path+"dataparallel/"+estimator+method+'_'+str((seed)*splits+i)+'.csv'), index_col=0)
            df = pd.concat([df, temp.iloc[:,-1]],axis=1)
        filename = path+"dataparallel/"+estimator+method+'_'+str(seed)+'_precombined'
        df.to_csv(filename+'.csv')
        outer_dict[estimator][method].append(filename)
        cmat = np.array(df.iloc[:,-splits:])
        with open(filename+'.npy', 'wb') as f:
            np.save(f, cmat)

def normal_run(path,cols,  n_seed, splits, method, estimator, runs,n_features):
    execute_feature_selection(path, runs, method,n_features)
    with open(path+"features/"+method+".txt","rb") as fp:
        features = pickle.load(fp)
    outer_dict = create_empty_dic(estimator, method)
    Parallel(n_jobs=-1)(delayed(execute_a_run)(path, i,runs[i],features[i], estimator, method) for  i  in range(len(runs)))
    combinefile(path, method,estimator,splits,n_seed, outer_dict)
    delete_interim_csv(estimator,method,len(runs))
    file_list = create_final_csv(path, outer_dict, n_seed, splits)

    if not estimator=='AllFeatures':
        featuresused=pd.DataFrame(columns=range(1,len(runs)+1))
        for  i  in range(len(runs)):
            featuresused.iloc[:,i]=pd.Series(cols[features[i]])
            featuresused.to_csv(path+"features/"+estimator+"_"+method+'_FEATURES.csv')   
    return file_list

def delete_interim_csv(estimator,method,runs_len):
    for i in range(runs_len):
        file = estimator+method+'_'+str(i)
        if os.path.exists(file+'.csv'):
            os.remove(file+'.csv')

def create_empty_dic(estimator, method):
    final_dict = {}
    final_dict[estimator] = {}
    final_dict[estimator][method] = []
    print(final_dict)
    return final_dict

def stringconverter(string_vector):
    return np.fromstring(string_vector[1:-1], dtype=np.int, sep=',')

def create_final_csv(path, outer_dict, n_seed, splits):
    final_names = []
    vfunc = np.vectorize(stringconverter)
    for estimator in outer_dict:
        for method in outer_dict[estimator]:
            dimension = pd.read_csv(outer_dict[estimator][method][0]+'.csv').shape
            final_df = pd.read_csv(outer_dict[estimator][method][0]+'.csv',index_col=0).iloc[:,range(dimension[1]-splits-1)]
            for file in outer_dict[estimator][method]:
                with open(file+'.npy', 'rb') as f:
                    a = np.load(f, allow_pickle=True)
                collapsed_array = []
                for i in range(len(a)):
                    array = np.array([vfunc(xi) for xi in a[i]])
                    array = np.sum(array,axis=0)
                    collapsed_array.append([array])
                final_df = pd.concat([final_df,pd.DataFrame(collapsed_array, columns=['Confusion Matrix '+file[-13]])], axis=1)
            final_filename = path+"resultsparallel/"+estimator+method+'_final'
            final_names.append(final_filename)
            final_df.to_csv(final_filename+'.csv')            
            cmat = np.array(final_df.iloc[:,-n_seed:])
            with open(final_filename+'.npy', 'wb+') as d:
                np.save(d, cmat)
    return final_names