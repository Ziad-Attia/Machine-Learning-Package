import glob, os
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter


def create_STATS(path):
    '''
    Gather the max accuracies and percisions of all the classifers and feature selection methods and save them in a csv file.

    Args:
        path: Directory from which the python file is being run
    '''
    os.chdir(path+'/resultsparallel')
    df = pd.DataFrame(columns=['filename','Accuracy', 'F1'])
    for file in glob.glob("*finalscore.csv"):
        data = pd.read_csv(file,skipinitialspace=True, header = 0)
        max_ac = data['Average Accuracy'].max()
        max_f1 = data['Average F1'].max()
        df = df.append({'filename':file,'Accuracy':max_ac,'F1': max_f1}, ignore_index=True)
    df.to_csv(path+'/STATS/max_scores_in_summary.csv',index=True)

def heatmap(path, title):
    '''
    Create heatmaps for the accuracies and percision scores of all the classifers and feature selection methods.

    Args:
        path: Directory from which the python file is being run
        title: the dependent variable of the dataset
    
    '''
    df = pd.read_csv(path+'/STATS/max_scores_in_summary.csv', index_col=[0])
    df['Accuracy'] = round(df['Accuracy'],3)
    df['F1'] = round(df['F1'],3)
    filenames = df["filename"]
    featureselection = []
    group = []
    classifier = []
    boostbag = []
    for name in filenames:  
        if "elasticnet" in name:
            classifier.append("EN")
        elif "knn" in name:
            classifier.append("KNN")
        elif "naive_bayes" in name:
            classifier.append("NB")
        elif "rdforest" in name:
            classifier.append("RF")
        elif "svm" in name:
            classifier.append("SVM")
        elif "xgboost" in name: 
            classifier.append("XGB")
        elif "logreg" in name: 
            classifier.append("LR")
            
        if "bag_finalscore" in name:
            boostbag.append("bag")
        elif "boost_finalscore" in name:
            boostbag.append("boost")
        else:  
            boostbag.append("none")
        
        if "cfs" in name:
            featureselection.append("CFS")
            group.append(1)
        elif "fcbf" in name:
            featureselection.append("FCBF")
            group.append(1)
        elif "mrmr" in name:
            if len(name.split('_'))==2:
                featureselection.append("MRMR")
            else:
                featureselection.append("MRMR-"+name.split('_')[1])
            group.append(1)
        elif "infogain" in name:
            featureselection.append("IG-"+name.split('_')[1])
            group.append(2)
        elif "reliefF" in name:
            featureselection.append("ReF-"+name.split('_')[1])
            group.append(2)
        elif "jmi" in name:
            if len(name.split('_'))==2:
                featureselection.append("JMI")
            else:
                featureselection.append("JMI-"+name.split('_')[1])
            group.append(1)
        elif "AllFeatures" in name:
            featureselection.append("All")
            group.append(0)
            
    df["Feature Selection"] = featureselection
    df["Classifier"] = classifier
    df["Boost or Bag"] = boostbag
    df["group"] = group
    df = df[df["Feature Selection"]!="notselected"]
    df = df.sort_values(by = 'group').reset_index().drop('index', axis=1)
    df.to_csv(path+'/STATS/'+"filenamesorted.csv", index=False)

    boost = df[df["Boost or Bag"] == "boost"]
    result = boost.pivot(index = ['group', 'Feature Selection'],columns = 'Classifier', values = 'Accuracy')
    result.reset_index(drop=True, inplace=True, level='group')
    if not result.empty:
        fig,ax = plt.subplots(figsize=(12,7))
        plt.xlabel("Feature Selection", fontsize = 25)
        plt.ylabel("Classifier", fontsize = 25)
        ax.tick_params(labelsize=15)
        ax.set_title(title, fontsize = 35)
        res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 22},linewidths=0.30,ax=ax )
        cbar = res.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 1))
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.locator_params(nbins=2, tight=True)
        cbar.ax.autoscale(enable=False)
        plt.tight_layout()
        plt.savefig(path+'/STATS/boost_Accuracy.png')
        plt.show()
        
        result = boost.pivot(index = ['group', 'Feature Selection'],columns = 'Classifier', values = 'F1')
        result.reset_index(drop=True, inplace=True, level='group')
        fig,ax = plt.subplots(figsize=(12,7))
        plt.xlabel("Feature Selection", fontsize = 25)
        plt.ylabel("Classifier", fontsize = 25)
        ax.tick_params(labelsize=15)
        ax.set_title(title, fontsize = 35)
        res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 22},linewidths=0.30,ax=ax )
        cbar = res.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 1))
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.locator_params(nbins=2, tight=True)
        cbar.ax.autoscale(enable=False)
        plt.tight_layout()
        plt.savefig(path+'/STATS/boost_F1.png')
        plt.show()

          
    bag = df[df["Boost or Bag"] == "bag"]
    result = bag.pivot(index = ['group', 'Feature Selection'],columns = 'Classifier', values = 'Accuracy')
    result.reset_index(drop=True, inplace=True, level='group')
    if not result.empty:
        fig,ax = plt.subplots(figsize=(12,7))
        plt.xlabel("Feature Selection", fontsize = 25)
        plt.ylabel("Classifier", fontsize = 25)
        ax.tick_params(labelsize=15)
        ax.set_title(title, fontsize = 35)
        res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 22},linewidths=0.30,ax=ax )
        cbar = res.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 1))
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.locator_params(nbins=2, tight=True)
        cbar.ax.autoscale(enable=False)
        plt.tight_layout()
        plt.savefig(path+'/STATS/bag_Accuracy.png')
        plt.show()

        result = bag.pivot(index = ['group', 'Feature Selection'],columns = 'Classifier', values = 'F1')
        result.reset_index(drop=True, inplace=True, level='group')
        fig,ax = plt.subplots(figsize=(12,7))
        plt.xlabel("Feature Selection", fontsize = 25)
        plt.ylabel("Classifier", fontsize = 25)
        ax.tick_params(labelsize=15)
        ax.set_title(title, fontsize = 35)
        res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 22},linewidths=0.30,ax=ax )
        cbar = res.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 1))
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.locator_params(nbins=2, tight=True)
        cbar.ax.autoscale(enable=False)
        plt.tight_layout()
        plt.savefig(path+'/STATS/bag_F1.png')
        plt.show()
        
    none = df[df["Boost or Bag"] == "none"]
    result = none.pivot(index = ['group', 'Feature Selection'],columns = 'Classifier', values = 'Accuracy')
    result.reset_index(drop=True, inplace=True, level='group')
    if not result.empty:
        fig,ax = plt.subplots(figsize=(12,7))
        plt.xlabel("Feature Selection", fontsize = 26)
        plt.ylabel("Classifier", fontsize = 26)
        ax.tick_params(labelsize=22)
        
        ax.set_title(title, fontsize = 32)
        res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 22},linewidths=0.30,ax=ax )
        cbar = res.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 1))
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.locator_params(nbins=2, tight=True)
        cbar.ax.autoscale(enable=False)
        plt.tight_layout()
        plt.savefig(path+'/STATS/noboostbag_Accuracy.png')
        plt.show()

        result = none.pivot(index = ['group', 'Feature Selection'],columns = 'Classifier', values = 'F1')
        result.reset_index(drop=True, inplace=True, level='group')
        fig,ax = plt.subplots(figsize=(12,7))
        plt.xlabel("Feature Selection", fontsize = 24)
        plt.ylabel("Classifier", fontsize = 24)
        ax.tick_params(labelsize=18)
        ax.set_title(title, fontsize = 32)
        res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 22},linewidths=0.30,ax=ax )
        cbar = res.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 1))
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.locator_params(nbins=2, tight=True)
        cbar.ax.autoscale(enable=False)
        plt.tight_layout()
        plt.savefig(path+'/STATS/noboostbag_F1.png')
        plt.show()