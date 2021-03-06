import gensim
import pandas as pd
import nltk
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

class Embedding_Word2Vec:
    # Variable generale pour le modele skyp-gram/CBOW
    
    """
    - n_size: ici corréspond à la taille de la couche caché qui va nous servir  comme étant le vecteur qui représente un mot.
    - n_window: est la distance maximale entre le mot actuel et les mots qui lui corréspondent.
    - n_workers: est le nombre de processeur que l'on souaite utiliser.
    - n_min_count: on ignore les mots dont la fréquence est inféireur à cette paramètre.
    - n_sg: nous permet de choisir entre skip-gram et cbow (par défaut skip-gram)
    - n_sh: nous permet de choisir entre la fonction de soft max et le négatif simpling lorsque l'on va faire une déscente de gradien pour la partie optimisation.(valeur par défaut soft max)

    """
    
    def __init__(self,n_window,n_min_count,n_workers,n_size=100,n_sg=1,n_hs=1):
        self.n_size=n_size
        self.n_window=n_window
        self.n_min_count=n_min_count
        self.n_workers=n_workers
        self.n_hs=n_hs
        self.n_sg=n_sg
        global model


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # Calculer la moyenne des mots à patie des vecteurs de mots.
    @staticmethod
    def __word_averaging(self,wv, words):
        all_words, mean = set(), []
    
        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            #logging.warning("ne peut pas calculer la similarité sans entrée %s", words)
            #  On enleve les mots dont la moyenne ne peut être calculer
            return np.zeros(wv.vector_size,)
        else:
         
            mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
            return mean
    ## Calcul sur l'ensemble du corpus
    @staticmethod
    def  __word_averaging_list(self,wv, text_list):
        return np.vstack([self.__word_averaging(self,wv, post) for post in text_list ])
    
    # on fit notre modèle sur X ( X_train par exemple)
    def fit(self,X,y=None):
        test_tokenized=None 
        if isinstance(X, pd.Series):
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))
        else:
            X=pd.DataFrame(X)
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))  
        
        self.model=gensim.models.Word2Vec(X_tokenized,size=self.n_size,window=self.n_window,min_count=self.n_min_count,
                                          workers=self.n_workers,sg=self.n_sg,hs=self.n_hs)
        self.model.init_sims(replace=True)
        
        return self
    # On transforme notre corpus grace à cette fonction en un matrice qui représente notre corpus
    def transform(self,X,y=None):
        test_tokenized=None 
        
        if isinstance(X, pd.Series):
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))
        else:
            X=pd.DataFrame(X)
            X_tokenized = list(X.apply(lambda r: gensim.utils.simple_preprocess(r)))
        wv=self.model.wv
      
        X_word_average = self.__word_averaging_list(self,self.model.wv,X_tokenized)
        scaler = MinMaxScaler()
        scaler.fit(X_word_average)
        XresMinMax=scaler.transform(X_word_average)    
        return(XresMinMax)
