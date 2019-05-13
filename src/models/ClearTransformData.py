import spacy
nlp = spacy.load('en')
import string
import numpy as np
from nltk.corpus import stopwords
import re


class Cleardataset:
    '''
    This class allow us to delete ponctuation, remove stopwords, to lower each word in the corpus
    and to make lemmatization. 

    '''

    def fit(self,X,y=None):
        return self

    
    def __fonction_nettoyage_text(self,df):
        # supprission des ponctuations
        rm_ponct = str.maketrans('','',string.punctuation)
        df = df.apply(lambda x:x.translate(rm_ponct))
    
        # suppression les unicodes
        df = df.apply(lambda x:x.encode("ascii","ignore").decode("utf-8"))
    
        # suppression des URLs
        df = df.apply(lambda x:re.sub(r'http\S+',"",x))
    
        # suppression des stopwords
        stop_en = stopwords.words("english")
        df = df.apply(lambda x:" ".join(x.lower() for x in np.str(x).split() if x.lower() not in stop_en))
        # Lemmatisation
        df = df.apply(lambda x:" ".join([ word.lemma_ for word in nlp(x) if word.lemma_!="-PRON-"]))              
    
        return(df)

    def transform(self,X,y=None):
        return self.__fonction_nettoyage_text(X)




    



