from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from Word2VecTransformer import Embedding_Word2Vec
from FastTextTransformer import FastTextTransformer
from sklearn.ensemble import RandomForestClassifier
from threading import Thread
from ClearTransformData import Cleardataset
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
import pandas as pd
import time
import pickle
import os


def calcul_prediction(modele_import, X, y):
    Accuracy, Precision, Recall, F1, MCC, cohen_kappa, name_clf, name_emb, Listtime = [], [], [], [], [], [], [], [], []
    for nb, model in enumerate(modele_import):
        get_name_model    =model.split("_")[-2:]
        get_name_model[1] =get_name_model[1].split(".")[0]
        name_emb[nb]      = get_name_model[0]
        name_clf[nb]      = get_name_model[1]
        Modele            = pickle.load(open(model, "rb"))
        t0 = time.time()
        y_pred            = Modele.predict(X)
        Listtime[nb]      = time.time()-t0
        Accuracy[nb]      = Modele.score(X,y)
        Precision[nb]     = precision_score(y_pred,y)
        Recall[nb]        = recall_score(y_pred,y)
        F1[nb]            = f1_score(y_pred,y)
        MCC[nb]           = matthews_corrcoef(y_pred,y)
        cohen_kappa[nb]   = cohen_kappa_score(y_pred,y)
    dict_res = {"Embeding":name_emb, 
              "Classifier" : name_clf,
              "Accuracy": Accuracy, 
              "Precision":Precision,
              "Recall":Recall,
              "F1":F1,
              "MCC":MCC,
              "cohen_kappa":cohen_kappa,
              "Time": Listtime
             }    
    return pd.DataFrame(dict_res)
        
    