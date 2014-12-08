import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import TruncatedSVD, ProjectedGradientNMF
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn import svm


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
    
            
if __name__ == '__main__':
    
    train_mat1 = np.loadtxt('train_mat8.txt', delimiter=",")
    test_mat1 = np.loadtxt('test_mat8.txt', delimiter=",")

    train_X = train_mat1[:,1:]
    test_X = test_mat1[:,1:]
    train_label = train_mat1[:,0]
    test_label = test_mat1[:,0]
            
    #Normalize the data
    normalizer = MinMaxScaler(feature_range=(0,1),copy=False)
    train_X = normalizer.fit_transform(train_X)
    test_X = normalizer.transform(test_X)
    
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_label)
    test_y = encoder.transform(test_label)
        
           
    print("training")
    k_features = [50, 100, 200, 300, 500]
       
    results = []
    methods = ['SVD', 'SRP', 'GRP', 'NMF']
        
    def updatescores(result, k, dimRed):
        """
        update parameters and scores
        """
        result.update({'dimRed' : dimRed, 'ndim' : k})
        results.append(result)

            
    for dimRed in methods:
        for k in k_features:
            print("select {} dimensions using {}".format(k, dimRed))
            if dimRed == 'SVD':
                train_X_sub, test_X_sub = select_features_svd(train_X, train_y,test_X, k)
            elif dimRed == 'SRP':       
                train_X_sub, test_X_sub = select_features_SparseRandomProjections(train_X, train_y,test_X, k)
            elif dimRed == 'GRP':             
                train_X_sub, test_X_sub = select_features_GaussianRandomProjections(train_X, train_y,test_X, k)
            elif dimRed == 'NMF':             
                train_X_sub, test_X_sub = select_features_nmf(train_X, train_y,test_X, k)
            elif dimRed == 'all':             
                train_X_sub, test_X_sub = train_X, test_X
                    
            print('Training SVM...')
            clf = svm.SVC(kernel='linear')
            result = benchmark(clf, train_X_sub, train_y, test_X_sub, test_y)
            updatescores(result, k, dimRed)            
            
    resultsDF = pd.DataFrame(results)
    resultsDF.to_csv('Emotient_data8_Results.csv')
    
