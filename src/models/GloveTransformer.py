import sys
import time
import os
import pandas as pd
sys.path.append('../')


from .gloveLocal.glove import build_vocab,build_cooccur,train_glove
from .gloveLocal.evaluate import make_id2word,merge_main_context
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

class GloveTransformer:

    def __init__(self,vector_size=100,iterations=25,window_size=10):
        self.vector_size=vector_size
        self.iterations=iterations
        self.window_size=window_size
        self.W=None
        self.vocab=None
        self.id2word=None

    def __word_representation(self,X):
    
        self.vocab = build_vocab(X)
        cooccur = build_cooccur(self.vocab,X, window_size=self.window_size)
        self.id2word = make_id2word(self.vocab)
        W_d = train_glove(self.vocab, cooccur, vector_size=self.vector_size, iterations=self.iterations)
        
        self.W= merge_main_context(W_d)
        return self
    
    def fit(self,X,y=None):
        return self.__word_representation(X)
        

    
    def __word_averaging(self, X):
        Corresp={self.id2word[i]:self.W[i,:] for i in range(self.W.shape[0])}
        result= [np.mean([Corresp[u] for u in x.split() if u in Corresp.keys()],axis=0) for x in X]
        result = [x if not np.isnan(np.sum(x)) else np.zeros(self.vector_size) for x in result]
        return np.asarray(result)
        
        
    def transform(self,X,y=None):
        X_tranform= self.__word_averaging(X)
        scaler = MinMaxScaler()
        scaler.fit( X_tranform)
        XresMinMax=scaler.transform(X_tranform)    
        return(XresMinMax)
    

    
    
        
if __name__=="__main__":
    
    print("Debut du programme")
    
    #testData=pd.read_csv("../../data/TestData.csv")
    print("Clear data")
    t1=time.time()
    
    
    if not os.path.exists("../../data/TrainDataClean.csv"):
        trainData=pd.read_csv("../../data/TrainData.csv")
        XtrainClean=Cleardataset().fit(trainData.resume).transform(trainData.resume)
        XtrainCleanPrim=Cleardataset().fit(trainData.description).transform(trainData.description)
        trainData.resume=XtrainClean
        trainData.description=XtrainCleanPrim
        trainData.to_csv("../../data/TrainDataClean.csv",index=False)
    else:
         trainData=pd.read_csv("../../data/TrainDataClean.csv")

    print("data cleared in:{}".format(time.time()-t1))
    testclass=GloveTransformer()
    
    #print(testclass.word_representation(test_corpus))
    res=testclass.fit(trainData.resume)
    
    pickle.dump( res, open( "../data/Embedding_dataText/resume_trainedDataGlove.sav", "wb" ) )
    
