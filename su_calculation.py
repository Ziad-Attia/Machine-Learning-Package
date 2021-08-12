import numpy as np

"""
utilities.py
Created by Prashant Shiralkar on 2015-02-06.
Utility methods to compute information-theoretic concepts such as 
entropy, information gain, symmetrical uncertainty
"""

def entropy(vec, base=2):
	"Returns the empirical entropy H(X) in the input vector"
	_, vec = np.unique(vec, return_counts=True)
	prob_vec = np.array(vec/float(sum(vec)))
	if base == 2:
		logfn = np.log2
	elif base == 10:
		logfn = np.log10
	else:
		logfn = np.log
	return prob_vec.dot(-logfn(prob_vec))

def conditional_entropy(x, y):
	"Returns H(X|Y)"
	uy, uyc = np.unique(y, return_counts=True)
	prob_uyc = uyc/float(sum(uyc))
	cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
	return prob_uyc.dot(cond_entropy_x)
	
def mutual_information(x, y):
	"Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y"
	return entropy(x) - conditional_entropy(x, y)

def su_calculation(x, y):
	"Returns 'symmetrical uncertainty' - a symmetric mutual information measure"
	return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))