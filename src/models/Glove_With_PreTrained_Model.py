import numpy as np
import pandas as pd
import time
import os
import pickle
from threading import Thread
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


"""
Ce fichier a pour usage d'entrainer un modèle avec un glove pre-entrainer
"""

Path_to_Glove_Trained = "../data/model_embedding_resum/glove.6B.300d.txt"
DIMGLOVE=300


model1 = [ ("clfMultinomialNB",MultinomialNB())]
param1={"alpha":[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]}
model2 = [("clfLogistic",LogisticRegression())]
param2={"C": [0.001,0.01,0.1,1,10,100,1000]}
model3 = [("clfSVM",LinearSVC())]
param3={"C": [0.1, 1, 10, 100, 1000]}
model4 = [("clfrandomForest",RandomForestClassifier())]
param4={"n_estimators": [200,500, 700]}
model5 = [ ("GradientBoosting",GradientBoostingClassifier())]
param5={"n_estimators": [20, 30, 40, 50, 60, 70, 80]}

modelfinal=[(model1,param1),(model2,param2),(model3,param3),(model4,param4),(model5,param5)]



def Embedd_vector(Path_to_Glove_Trained=Path_to_Glove_Trained):
    
    f = open(Path_to_Glove_Trained,"r")
    dict_Glove={}
    for x in f:
        u = x.split()
        dict_Glove[u[0]]=list(np.array(u[1:],dtype=np.float32))

    return dict_Glove

def transformX(X,Path_to_Glove_Trained=Path_to_Glove_Trained):
    
    Corresp= Embedd_vector(Path_to_Glove_Trained=Path_to_Glove_Trained)    
    result = [np.mean([Corresp[u] for u in x.split() if u in Corresp.keys()],axis=0) for x in X]
    
    result = [x if not np.isnan(np.sum(x)) else np.zeros(self.vector_size) for x in result]

    X_tranform = np.asarray(result)
    scaler = MinMaxScaler()
    scaler.fit(X_tranform)
    XresMinMax=scaler.transform(X_tranform)
    
    return(XresMinMax)


def trainModelGlove(X,Y,model,param,path,cv,n_jobs,Path_to_Glove_Trained=Path_to_Glove_Trained):
    print("Modelx {}".format(model))
    modele  = GridSearchCV(estimator = model[0][1], param_grid = param, cv=cv,n_jobs=n_jobs)
    print("Model lauch:{}_{}".format(model,param))
    t0=time.time()

    X_transformed = transformX(X,Path_to_Glove_Trained=Path_to_Glove_Trained)
    modele.fit(X_transformed,Y)
    print("Model lauched successfull. Execution times:{}".format(time.time()-t0))
    word_transformer = "Glove"
    name_model = model[0][0]
    modele_name = "model_{}_{}.sav".format(word_transformer,name_model)
    pickle.dump( modele, open( path+modele_name, "wb" ) )

 





class BestModelFinder(Thread):

    """Model with threading to parallelize your luncher."""

    def __init__(self,X,Y,ListParamModel,path,cv=5,n_jobs=2,Path_to_Glove_Trained=Path_to_Glove_Trained):
        Thread.__init__(self)
        self.X=X
        self.Y=Y
        self.ListParamModel = ListParamModel
        self.path=path
        self.cv=cv
        self.n_jobs=n_jobs
        self.Path_to_Glove_Trained=Path_to_Glove_Trained

    def run(self):
        
        for mod,par in self.ListParamModel:
            trainModelGlove(X=self.X,Y=self.Y,model=mod,param=par,path=self.path,cv=self.cv,n_jobs=self.n_jobs,Path_to_Glove_Trained=self.Path_to_Glove_Trained)


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

    path_to_modele="../data/"
    demiLen=int(len(modelfinal)/2)
    modelList1 = [modelfinal[i] for i in range((demiLen))]
    modelList2 = [modelfinal[i] for i in range((demiLen),len(modelfinal))]

    
    Thread1 = BestModelFinder(X= trainData.description,Y= trainData.Labels,ListParamModel=modelList1,path=path_to_modele)
    Thread2 = BestModelFinder(X= trainData.description,Y= trainData.Labels,ListParamModel=modelList2,path=path_to_modele)

    print("Execute and save modele:")

    Thread1.start()
    Thread2.start()

    Thread1.join()
    Thread2.join()
