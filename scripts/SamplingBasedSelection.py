import numpy as np
import scipy.sparse

class WBS():
    
    def __init__(self, data, testData, r=10):
        self.data = data
        self.testData = testData
        self._crank=r
       
    def sample(self, s, probs):        
        prob_rows = np.cumsum(probs.flatten())            
        temp_ind = np.zeros(s, np.int32)
   
        for i in range(s):     
            v = np.random.rand()       
            try:
                tempI = np.where(prob_rows >= v)[0]
                temp_ind[i] = tempI[0]        
            except:
                temp_ind[i] = len(prob_rows)
           
        return np.sort(temp_ind)
       
    def sample_probability(self):
       
        dsquare = self.data.multiply(self.data)    
        pcol = np.array(dsquare.sum(axis=0), np.float64)
        pcol /= pcol.sum()    
        return pcol.reshape(-1,1)
                           
    def computeWBS(self):                
        self._C = self.data[:, self._cid] * scipy.sparse.csc_matrix(np.diag(self._ccnt)) 
            
    def factorize(self):
        pcol = self.sample_probability()
        self.data = scipy.sparse.vstack([self.data, self.testData])
        self._cid = self.sample(self._crank, pcol)
        self._ccnt = np.ones(len(self._cid))                             
        self.computeWBS()
   
   
class SS():
    
    def __init__(self, data, svdV, r=10, svdRank=1):
        self.data = data
        self._crank = r
        self.svdRank = svdRank
        self.svdV = svdV

       
    def sample(self, s, probs):        
        prob_rows = np.cumsum(probs.flatten())            
        temp_ind = np.zeros(s, np.int32)
   
        for i in range(s):     
            v = np.random.rand()
                       
            try:
                tempI = np.where(prob_rows >= v)[0]
                temp_ind[i] = tempI[0]        
            except:
                temp_ind[i] = len(prob_rows)
           
        return np.sort(temp_ind)
       
    def sample_probability(self):
        
        self.svdReduced = self.svdV[range(self.svdRank),:]
        self.svdReduced = np.square(self.svdReduced) 
        pcol = np.array(self.svdReduced.sum(axis=0), np.float64)/self.svdRank
        return pcol.reshape(-1,1)
                           
    def computeSS(self):                
        self._C = self.data[:, self._cid] * scipy.sparse.csc_matrix(np.diag(self._ccnt)) 
            
    def factorize(self):
        pcol = self.sample_probability()
        self._cid = self.sample(self._crank, pcol)
        self._ccnt = np.ones(len(self._cid))                             
        self.computeSS()