#!/usr/bin/env python
from __future__ import division
import re
import numpy as np
import pandas as pd
from scipy.sparse import vstack, csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD, ProjectedGradientNMF
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn import svm
from sklearn.lda import LDA
from nltk import stem
from nltk.corpus import stopwords
from SubspaceSampling import WBS, SS

class StemTokenizer(object):
    """
    Tokenizer for CountVectorizer with stemming support
    """
    def __init__(self):
        self.wnl = stem.WordNetLemmatizer()
        self.word = re.compile('[a-zA-Z]+')
        
    def __call__(self, doc):
        tokens = re.split('\W+', doc.lower())
        tokens = [self.wnl.lemmatize(t) for t in tokens]
        return tokens
            
def build_tfidf(train_data, test_data):
    stops = stopwords.words('english')
    counter = CountVectorizer(tokenizer=StemTokenizer(),
                              stop_words=stops, min_df=3,
                              dtype=np.double)
    counter.fit(train_data)
    train_tf = counter.transform(train_data)
    test_tf = counter.transform(test_data)
    transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    train_tfidf = transformer.fit_transform(train_tf)
    test_tfidf = transformer.transform(test_tf)
    return train_tfidf, test_tfidf

def benchmark(clf, train_X, train_y, test_X, test_y):
    """
    evaluate classification
    """
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
        
    f1 = metrics.f1_score(test_y, pred, average='weighted')
    recall = metrics.recall_score(test_y, pred, average='weighted')
    precision = metrics.precision_score(test_y, pred, average='weighted')
    accuracy = metrics.accuracy_score(test_y, pred)
    result = {'f1' : f1, 'recall' : recall, 'precision' : precision, 'accuracy' : accuracy}
    return result
        
def select_features_chi2(train_X, train_y, test_X, k):
    if k == 'all':
        return train_X, test_X
                
    selector = SelectKBest(chi2, k=k)
    selector.fit(train_X, train_y)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X
    
def select_features_svd(train_X, train_y, test_X, k):
    selector = TruncatedSVD(n_components=k, random_state=42)
    selector.fit(train_X)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X
    
def select_features_nmf(train_X, train_y, test_X, k):
    selector = ProjectedGradientNMF(n_components=k, init='nndsvd', random_state=42)
    selector.fit(train_X)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X
    
def select_features_SparseRandomProjections(train_X, train_y, test_X, k):
    selector = SparseRandomProjection(n_components=k, random_state=42)
    selector.fit(train_X)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X

def select_features_GaussianRandomProjections(train_X, train_y, test_X, k):
    selector = GaussianRandomProjection(n_components=k, random_state=42)
    selector.fit(train_X)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X
    
def select_features_LDA(train_X, train_y, test_X, k):
    selector = LDA(n_components=k)
    selector.fit(train_X.toarray(), train_y)
    train_X = selector.transform(train_X.toarray())
    test_X = selector.transform(test_X.toarray())
    return train_X, test_X
    
def select_features_WBS(train_X, train_y, test_X, k):
    trainSamples = train_X.shape[0]
    testSamples = test_X.shape[0]
    selector = WBS(train_X, test_X, r=k)
    selector.factorize()
    train_X = selector._C[range(trainSamples),:]
    test_X = selector._C[range(trainSamples,trainSamples+testSamples),:]
    return train_X, test_X

def select_features_SS(train_X, train_y, test_X, k, svdRank, svdV):
    trainSamples = train_X.shape[0]
    testSamples = test_X.shape[0]
    selector = SS(vstack([train_X, test_X]), r=k, svdRank = svdRank, svdV = svdV)
    selector.factorize()
    train_X = selector._C[range(trainSamples),:]
    test_X = selector._C[range(trainSamples,trainSamples+testSamples),:]
    return train_X, test_X
    
def InfoEntropy(results):
    import math
    nInstances = float(sum(results)) 
    if nInstances == 0: 
        return 0 
    probs = results/nInstances 
    t = np.choose(np.greater(probs,0.0),(1,probs)) 
    return sum(-probs*np.log(t)/math.log(2)) 
    
def InfoGain(varMat): 
    variableRes = np.sum(varMat,0)
    overallRes = np.sum(varMat,1)
    term2 = 0 
  
    for i in xrange(len(variableRes)): 
        term2 = term2 + variableRes[i] * InfoEntropy(varMat[i])  
    tSum = sum(overallRes) 
    if tSum != 0.0: 
        term2 = 1./tSum * term2 
        gain = InfoEntropy(overallRes) - term2 
    else: 
        gain = 0 
    return gain 

def IG (train_X, train_y):
    InfGain = []
    for term in range(train_X.shape[1]):
        col = train_X[:,term].toarray()
        col[col > 0] = 1
        DF = pd.DataFrame(np.vstack([[item for sublist in col for item in sublist], train_y]))
        DF = DF.transpose()
        DF.columns = ['A','B']
        DF = DF.pivot_table(rows='A', columns='B', aggfunc=len, fill_value = 0)
        InfGain.append(InfoGain(DF))
    
    return np.asarray(InfGain)
        
def select_features_IG(train_X, test_X, InfGain, k):
    mask = np.zeros(InfGain.shape, dtype=bool)   
    mask[np.argsort(InfGain, kind="mergesort")[-k:]] = 1
    train_X = train_X[:,mask]
    test_X = test_X[:,mask]
    
    return train_X, test_X
    
def MutInf(varMat): 
    variableRes = np.sum(varMat,0) 
    overallRes = np.sum(varMat,1)
    MItc = -np.Infinity
    pt = 1.0*overallRes[1]/overallRes[0]
    for docClass in range(len(variableRes)):        
        pc = 1.0*variableRes[docClass]/sum(variableRes)
        ptc = 1.0*varMat.iloc[1,docClass]/sum(variableRes)
        classMI = np.log2(ptc/(pt*pc))
        if classMI> MItc:
            MItc = classMI
    return MItc 
    
def MI (train_X, train_y):
    MutualInformation = []
    for term in range(train_X.shape[1]):
        col = train_X[:,term].toarray()
        col[col > 0] = 1
        if (col.sum()==train_X.shape[0]):
            MI = -np.Infinity
            MutualInformation.append(MI)
        else:
            DF = pd.DataFrame(np.vstack([[item for sublist in col for item in sublist], train_y]))
            DF = DF.transpose()
            DF.columns = ['A','B']
            DF = DF.pivot_table(rows='A', columns='B', aggfunc=len, fill_value = 0)
            MutualInformation.append(MutInf(DF))
    
    return np.asarray(MutualInformation)

def select_features_MI(train_X, test_X, MutInf, k):
    mask = np.zeros(MutInf.shape, dtype=bool)   
    mask[np.argsort(MutInf, kind="mergesort")[-k:]] = 1
    train_X = train_X[:,mask]
    test_X = test_X[:,mask]
    
    return train_X, test_X
    
def GI(varMat): 
    variableRes = np.sum(varMat,0) 

    pc = 1.0*variableRes/sum(variableRes)
    pct = 1.0*varMat.iloc[1,:]/sum(varMat.iloc[1,:])
    
    denominator = sum(pct/pc)
    Pnew = (pct/pc)/denominator
    PctWeighted = sum(Pnew**2)
    
    return PctWeighted 
    
def GiniIndex (train_X, train_y):
    GiniIndex = []
    for term in range(train_X.shape[1]):
        col = train_X[:,term].toarray()
        col[col > 0] = 1
        if (col.sum()==train_X.shape[0]):
            GinInd = 0
            GiniIndex.append(GinInd)
        else:
            DF = pd.DataFrame(np.vstack([[item for sublist in col for item in sublist], train_y]))
            DF = DF.transpose()
            DF.columns = ['A','B']
            DF = DF.pivot_table(rows='A', columns='B', aggfunc=len, fill_value = 0)
            GiniIndex.append(GI(DF))
    
    return np.asarray(GiniIndex)
    
def select_features_GI(train_X, test_X, GiniIndex, k):
    mask = np.zeros(GiniIndex.shape, dtype=bool)   
    mask[np.argsort(GiniIndex, kind="mergesort")[-k:]] = 1
    train_X = train_X[:,mask]
    test_X = test_X[:,mask]
    
    return train_X, test_X
    
if __name__ == '__main__':

    dataset = "newsgroup"    
    
    if dataset == "reuters52" :
        
        trainData = pd.ExcelFile('Reuters-21578 R52 train.xlsx')
        testData = pd.ExcelFile('Reuters-21578 R52 test.xlsx')
        trainData = trainData.parse('Sheet1', header=None)
        testData = testData.parse('Sheet1', header=None)
        
        train_text = trainData.iloc[:,1]
        test_text = testData.iloc[:,1]
        train_label = trainData.iloc[:,0]
        test_label = testData.iloc[:,0]
        
    elif dataset == "reuters8":
        trainData = pd.ExcelFile('Reuters-21578 R8 train.xlsx')
        testData = pd.ExcelFile('Reuters-21578 R8 test.xlsx')
        trainData = trainData.parse('Sheet1', header=None)
        testData = testData.parse('Sheet1', header=None)
        
        train_text = trainData.iloc[:,1]
        test_text = testData.iloc[:,1]
        train_label = trainData.iloc[:,0]
        test_label = testData.iloc[:,0]
        
    elif dataset == "newsgroup":
        trainData = pd.ExcelFile('newsgroup train.xlsx')
        testData = pd.ExcelFile('newsgroup test.xlsx')
        trainData = trainData.parse('Sheet1', header=None)
        testData = testData.parse('Sheet1', header=None)
        
        train_text = trainData.iloc[:,1]
        test_text = testData.iloc[:,1]
        train_label = trainData.iloc[:,0]
        test_label = testData.iloc[:,0]
            
    print('encode labels')
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_label)
    test_y = encoder.transform(test_label)
    
    print("build tfidf")
    train_X, test_X = build_tfidf(train_text, test_text)

    print("calculating Information Gain")
    InfGain = IG(train_X, train_y)
    
    print("calculating Mutual Information")
    MutInfor = MI(train_X, train_y)
    
    print("calculating Gini Index")
    GinInd = GiniIndex(train_X, train_y)
    
    print("calculating SVD for Subspace-Sampling")
    U, S, V = np.linalg.svd(train_X.todense(),full_matrices =False)    

    train_X = csr_matrix(train_X)
    test_X = csr_matrix(test_X)

    print("training")
       
    k_features = [2, 5, 10, 50, 100, 200, 500, 1000]
    
    results = []
    methods = ['IG','chi-2', 'MImax', 'GI', 'SVD', 'Sparse Random Projection', 'Gaussian Random Projection', 'NMF', 'LDA', 'WBS','SS']
    
    def updatescores(name, k, result):
        """
        update parameters and scores
        """
        result.update({'method' : name , 'n_features' : k})
        results.append(result)
    

    for method in methods:        
        for k in k_features:
            print("select {} features using {}".format(k, method))
            
            if method == 'chi-2':
                train_X_sub, test_X_sub = select_features_chi2(train_X, train_y,test_X, k)
            elif method == 'IG':
                train_X_sub, test_X_sub = select_features_IG(train_X, test_X, InfGain, k)
            elif method == 'MImax':
                train_X_sub, test_X_sub = select_features_MI(train_X, test_X, MutInfor, k)
            elif method == 'GI':
                train_X_sub, test_X_sub = select_features_GI(train_X, test_X, GinInd, k)
            elif method == 'SVD':
                train_X_sub, test_X_sub = select_features_svd(train_X, train_y,test_X, k)
            elif method == 'Sparse Random Projection':             
                train_X_sub, test_X_sub = select_features_SparseRandomProjections(train_X, train_y,test_X, k)
            elif method == 'Gaussian Random Projection':             
                train_X_sub, test_X_sub = select_features_GaussianRandomProjections(train_X, train_y,test_X, k)
            elif method == 'NMF':             
                train_X_sub, test_X_sub = select_features_nmf(train_X, train_y,test_X, k)
            elif method == 'LDA':             
                if k < len(np.unique(train_y)):                
                    train_X_sub, test_X_sub = select_features_LDA(train_X, train_y,test_X, k)     
                else:
                    continue
            elif method == 'WBS':             
                train_X_sub, test_X_sub = select_features_WBS(train_X, train_y,test_X, k)
            elif method == 'SS':             
                train_X_sub, test_X_sub = select_features_SS(train_X, train_y,test_X, k, svdRank=100, svdV=V)
            elif method == 'all':             
                train_X_sub, test_X_sub = train_X, test_X
                    
            print('Training SVM...')
            clf = svm.LinearSVC()
            result = benchmark(clf, train_X_sub, train_y, test_X_sub, test_y)
            updatescores(method, k, result)
            
            
    resultsDF = pd.DataFrame(results)
    print resultsDF 
    resultsDF.to_csv('Results.csv')