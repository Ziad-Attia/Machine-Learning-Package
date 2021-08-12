import pandas as pd
import numpy as np
import statistics
import math

def accuracy(tp, tn, total):
    '''
    Calculates the accuracy from the confusion matrix.

    Args:
        tp: True Positive
        tn: True Negative
        total: Total Population
    
    Returns the accuracy
    '''
    return (tp+tn)/total

def precision(tp, fp):
    '''
    Calculates the precision from the confusion matrix.

    Args:
        tp: True Positive
        fp: False Positive
    
    Returns the precision
    '''
    if tp+fp==0:
        return 0
    return (tp/(tp+fp))

def recall(tp, fn):
    '''
    Calculates the recall from the confusion matrix.

    Args:
        tp: True Positive
        fn: False Negative
    
    Returns the recall
    '''
    if tp+fn ==0:
        return 0
    return (tp/(tp+fn))

def f1(precision, recall):
    '''
    Calculates the f1 score from the precision and recall.

    Args:
        precision
        recall
    
    Returns the recall
    '''
    if precision+recall ==0:
        return 0
    return 2*((precision*recall)/(precision+recall))

def score(files, n_seed):
    '''
    Calculates the Average Accuracy, Average Precision, Average Recall, Average F1, and Standard Error of Accuracy 
    from all the validation runs of each classifier and feature selection method and saves them in a csv.

    Args:
        files: Arraylike structure that contains all the names of all the classifier-feature selection combinations
        n_seed: Number of validations
    '''
    for filename in files:
        df = pd.read_csv(filename+'.csv', index_col=0)
        with open(filename+'.npy', 'rb') as f:
            confusion_array = np.load(f, allow_pickle=True)
        average_array = np.zeros((len(confusion_array),5))
        for i in range(len(confusion_array)):
            acc = []
            prec = []
            rec = []
            f = []
            for j in range(n_seed):
                tn = confusion_array[i][j][0]
                fp = confusion_array[i][j][1]
                fn = confusion_array[i][j][2]
                tp = confusion_array[i][j][3]
                acc.append(accuracy(tp, tn, sum([tn,fp,fn,tp])))
                p = precision(tp, fp)
                prec.append(p)
                r = recall(tp, fn)
                rec.append(r)
                f.append(f1(p, r))
            if(n_seed>1): 
                average_array[i] = [sum(acc)/n_seed,sum(prec)/n_seed,sum(rec)/n_seed,sum(f)/n_seed, 2*(math.sqrt(statistics.variance(acc))/math.sqrt(n_seed))]
            else:
                average_array[i] = [sum(acc)/n_seed,sum(prec)/n_seed,sum(rec)/n_seed,sum(f)/n_seed, np.nan]
        averages = pd.DataFrame(average_array, columns=['Average Accuracy','Average Precision','Average Recall','Average F1', 'Standard Error of Accuracy'])
        final_df = pd.concat([df.iloc[:,range(df.shape[1])], averages],axis=1)
        final_df.to_csv(filename+'score.csv',index=True)











    
