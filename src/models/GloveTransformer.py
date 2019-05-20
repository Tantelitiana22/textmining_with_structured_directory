import sys
sys.path.append('..')


from gloveLocal.glove import build_vocab,build_cooccur,train_glove
from gloveLocal.evaluate import make_id2word,merge_main_context
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
        return np.asarray(result)
        
        
    def transform(self,X,y=None):
        X_tranform= self.__word_averaging(X)
        scaler = MinMaxScaler()
        scaler.fit( X_tranform)
        XresMinMax=scaler.transform(X_tranform)    
        return(XresMinMax)
    

    
    
        
if __name__=="__main__":
    
    test_corpus = ("""human interface computer                                                                                                                                                            
    survey user computer system response time                                                                                                                                                             
    eps user interface system                                                                                                                                                                             
    system human system eps                                                                                                                                                                               
    user response time                                                                                                                                                                                    
    trees                                                                                                                                                                                                 
    graph trees                                                                                                                                                                                           
    graph minors trees                                                                                                                                                                                    
    graph minors survey                                                                                                                                                                                   
    I like graph and stuff                                                                                                                                                                                
    I like trees and stuff                                                                                                                                                                                
    Sometimes I build a graph                                                                                                                                                                             
    Sometimes I build trees""").split("\n")

    testclass=GloveTransformer()
    
    #print(testclass.word_representation(test_corpus))
    print(testclass.fit(test_corpus).transform(test_corpus))
