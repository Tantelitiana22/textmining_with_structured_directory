import warnings
warnings.filterwarnings('ignore')
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
from sklearn.ensemble import GradientBoostingClassifier
from GloveTransformer import GloveTransformer
import pandas as pd
import time
import pickle
import os


PathToGloveTrained = "../data/model_embedding_resum/resume_trainedDataGlove.sav"

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

class BestModelFinder(Thread):

    """Model with threading to parallelize your luncher."""

    def __init__(self,X,Y,ListParamModel,path,cv=5,n_jobs=2):
        Thread.__init__(self)
        self.X=X
        self.Y=Y
        self.ListParamModel = ListParamModel
        self.path=path
        self.cv=cv
        self.n_jobs=n_jobs

    def __ApplyModel(self,model,param):
        
        gloveModel = pickle.load(open(PathToGloveTrained,"rb"))
        X_transformed = gloveModel.transform(self.X)

        modele  = GridSearchCV(estimator = model[0][1], param_grid = param, cv=self.cv,n_jobs=self.n_jobs)
        print("Model lauch:{}_{}".format(model,param))
        t0=time.time()
        modele.fit(X_transformed,self.Y)
        print("Model lauched successfull. Execution times:{}".format(time.time()-t0))
        word_transformer="Glove"
        name_model=model[0][0]
        modele_name="model_resume_{}_{}.sav".format(word_transformer,name_model)
        pickle.dump( modele, open( self.path+modele_name, "wb" ) )

    def run(self):
        cleardata=Cleardataset()
        for mod,par in self.ListParamModel:
            self.__ApplyModel(model=mod,param=par)
            
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
   	    
    path_to_modele="../data/model_with_resum/model_with_resum"
    demiLen=int(len(modelfinal)/2)
    modelList1 = [modelfinal[i] for i in range((demiLen))]
    modelList2 = [modelfinal[i] for i in range((demiLen),len(modelfinal))]

   
    Thread1 = BestModelFinder(X= trainData.resume,Y= trainData.Labels,ListParamModel=modelList1,path=path_to_modele)
    Thread2 = BestModelFinder( trainData.resume,Y= trainData.Labels,ListParamModel=modelList2,path=path_to_modele)

    print("Execute and save modele:")
    Thread1.start()
    Thread2.start()

    Thread1.join()
    Thread2.join()

