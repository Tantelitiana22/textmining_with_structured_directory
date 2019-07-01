from ModelDeepLearning import ModelDeepLearning
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
from GloveTransformer import GloveTransformer
from Glove_With_PreTrained_Model import transformX
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score,accuracy_score
import pandas as pd
import time
import pickle
import os
import glob2

def calcul_prediction(modele_import, X, y):
    Accuracy, Precision, Recall, F1, MCC, cohen_kappa, name_clf, name_emb, Listtime = [], [], [], [], [], [], [], [], []
    for nb, model in enumerate(modele_import):
        get_name_model    =model.split("_")[-2:]
        get_name_model[1] =get_name_model[1].split(".")[0]
        name_emb.append(get_name_model[0])
        name_clf.append( get_name_model[1])
        print("Model at {}".format(model))
        Modele            = pickle.load(open(model, "rb"))
        t0 = time.time()
        y_pred            = Modele.predict(X)
        Listtime.append(time.time()-t0)
        Accuracy.append( accuracy_score(y_pred,y))
        Precision.append(precision_score(y_pred,y,average= "weighted"))
        Recall.append(recall_score(y_pred,y,average='weighted'))
        F1.append(f1_score(y_pred,y,average='weighted'))
        MCC.append(matthews_corrcoef(y_pred,y))
        cohen_kappa.append(cohen_kappa_score(y_pred,y))
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
        

if __name__=="__main__":

    modelList = glob2.glob('../data/*.sav')
    modelGLove = [x for x in modelList if "Glove" in x.split("_")]
    modelNormal = [x for x in modelList if "Glove" not in  x.split("_")]

    print(modelGLove)
    print("modele{}".format(modelList))
    
    modelListResum = glob2.glob('../data/model_with_resum/*.sav')
    modelGLoveResum = [x for x in modelListResum if "Glove" in  x.split("_")]
    modelNormalResum = [x for x in modelListResum if "Glove" not in  x.split("_")]

    print(modelGLoveResum)


    if not os.path.exists("../../data/TestDataClean.csv"):

        TestData=pd.read_csv("../../data/TestData.csv")
        TestDataClean=Cleardataset().fit(TestData.description).transform(TestData.description)
        TestDataClean1=Cleardataset().fit(TestData.description).transform(TestData.resume)
        TestData.description=TestDataClean
        TestData.resume = TestDataClean1
        trainData.to_csv("../../data/TestDataClean.csv",index=False)
        
    else:
        TestData=pd.read_csv("../../data/TestDataClean.csv")

    t1 = time.time()
    XtranformGlovePretrained = transformX(TestData.description)
    print("Glove transformer duration: {}".format(time.time()-t1))
    dfresGlove =calcul_prediction(modelGLove,X=XtranformGlovePretrained , y = TestData.Labels)
    dfresNormal=calcul_prediction(modelNormal,X=TestData.description,y=TestData.Labels)

    frame = [dfresGlove,dfresNormal]
    df = pd.concat(frame)
    df.to_csv("../data/performance_model_evaluation.csv",index= False)

    
    gloveEmbedding =  pickle.load(open("../data/model_embedding_resum/resume_trainedDataGlove.sav","rb"))
    XresumGloveTransform = gloveEmbedding.transform(TestData.resume)
    dfresGloveResume = calcul_prediction(modelGLoveResum,X= XresumGloveTransform,y=TestData.Labels)
    dfresResume = calcul_prediction(modelNormalResum,X = TestData.resume, y = TestData.Labels)

    
   

    frameResum =  [dfresGloveResume , dfresResume ]
    dfresum = pd.concat(frameResum )
    dfresum.to_csv("../data/model_with_resum/performance_model_evaluation_resum.csv",index=False)


    
