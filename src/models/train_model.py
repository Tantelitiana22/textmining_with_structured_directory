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
import pandas as pd
import time
import pickle
import os

HOME_VAR="/home/arakotoarijaona/Bureau/arakotoarijaona/"

model2 = [("tfidf",TfidfVectorizer()) , ("clfMultinomialNB",MultinomialNB())]
param2={"clfMultinomialNB__alpha":[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]}
model1 = [("tfidf",TfidfVectorizer()) , ("clfLogistic",LogisticRegression())]
param1={"clfLogistic__C": [0.001,0.01,0.1,1,10,100,1000]}
model6 = [("tfidf",TfidfVectorizer()) , ("clfSVM",LinearSVC())]
param6={"clfSVM__C": [0.1, 1, 10, 100, 1000]}
model9 = [("tfidf",TfidfVectorizer()) , ("clfrandomForest",RandomForestClassifier())]
param9={"clfrandomForest__n_estimators": [200,500, 700]}



model = [("w2v", Embedding_Word2Vec(n_size = 300,n_window = 5,n_min_count = 10,n_workers = 4)), ("clfLogistic",LogisticRegression())]
param={"w2v__model":["cbow","skipgram"],"clfLogistic__C": [0.001,0.01,0.1,1,10,100,1000]}
model3 = [("w2v", Embedding_Word2Vec(n_size = 300,n_window = 5,n_min_count = 10,n_workers = 4)),("clfMultinomialNB",MultinomialNB())]
param3={"w2v__model":["cbow","skipgram"],"clfMultinomialNB__alpha":[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]}
model7 = [("w2v", Embedding_Word2Vec(n_size = 300,n_window = 5,n_min_count = 10,n_workers = 4)),("clfSVM",LinearSVC())]
param7={"w2v__model":["cbow","skipgram"],"clfSVM__C":[0.1, 1, 10, 100, 1000]}
model10 = [("w2v", Embedding_Word2Vec(n_size = 300,n_window = 5,n_min_count = 10,n_workers = 4)),("clfrandomForest",RandomForestClassifier())]
param10={"w2v__model":["cbow","skipgram"],"clfrandomForest__n_estimators":[200,500,700]}




ft_home =HOME_VAR+'textmining_with_structured_directory/data/fastText-0.2.0/fasttext'
Input=HOME_VAR+"textmining_with_structured_directory/src/models/FastTestFolder/xtrain.txt"

model4 = [("fastText",FastTextTransformer(inputFile=Input,ft_home=ft_home,size=300)),("clfMultinomialNB",MultinomialNB())]
param4={"fastText__model":["cbow","skipgram"],"clfMultinomialNB__alpha":[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]}
model5 = [("fastText",FastTextTransformer(inputFile=Input,ft_home=ft_home,size=300)) ,("clfLogistic",LogisticRegression())]
param5={"fastText__model":["cbow","skipgram"],"clfLogistic__C": [0.001,0.01,0.1,1,10,100,1000]}
model8 = [("fastText",FastTextTransformer(inputFile=Input,ft_home=ft_home,size=300)) ,("clfSVM",LinearSVC())]
param8={"fastText__model":["cbow","skipgram"],"clfSVM__C": [0.1, 1, 10, 100, 1000]}
model11 = [("fastText",FastTextTransformer(inputFile=Input,ft_home=ft_home,size=300)),("clfrandomForest",RandomForestClassifier())]
param11={"fastText__model":["cbow","skipgram"],"clfrandomForest__n_estimators":[200,500,700]}





modelfinal=[(model,param),(model1,param1),(model2,param2),(model3,param3),(model4,param4),
            (model5,param5),(model6,param6),(model7,param7),(model8,param8),(model9,param9),
            (model10,param10),(model11,param11)]




# def ma_fonction(X, Y, model, param, path=path_to_modele, cv = cv, n_jobs=3):
#     pipeline = Pipeline(model)
#     modele  = GridSearchCV(estimator = pipeline, param_grid = param, cv = cv, n_jobs=n_jobs)
#     modele.fit(X,Y)
#     modele_name="model_{}_{}.sav".format(model,param)
#     pickle.dump( modele, open( path+modele_name, "wb" ) )
      
# for mod,par in modelfinal:
#     ma_fonction(x_train,y_train,model=mod,param=par,path)



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

    def __ApplyModel(self,X, Y,model,param):
        pipeline = Pipeline(model)
        modele  = GridSearchCV(estimator = pipeline, param_grid = param, cv=self.cv,n_jobs=self.n_jobs)
        print("Model lauch:{}_{}".format(model,param))
        t0=time.time()
        modele.fit(X,Y)
        print("Model lauched successfull. Execution times:{}".format(time.time()-t0))
        word_transformer=model[0][0]
        name_model=model[1][0]
        modele_name="model_{}_{}.sav".format(word_transformer,name_model)
        pickle.dump( modele, open( self.path+modele_name, "wb" ) )

    def run(self):
        cleardata=Cleardataset()
        for mod,par in self.ListParamModel:
            self.__ApplyModel(self.X,self.Y,model=mod,param=par)



if __name__=="__main__":
    print("Debut du programme")
    
    #testData=pd.read_csv("../../data/TestData.csv")
    print("Clear data")
    t1=time.time()
    
    if not os.path.exists("../../data/TrainDataClean.csv"):
        trainData=pd.read_csv("../../data/TrainData.csv")
        XtrainClean=Cleardataset().fit(trainData.description).transform(trainData.description)
        trainData.description=XtrainClean
        trainData.to_csv("../../data/TrainDataClean.csv",index=False)
    else:
        XtrainClean=pd.read_csv("../../data/TrainDataClean.csv")

    print("data cleared in:{}".format(time.time()-t1))
   	    
    path_to_modele=HOME_VAR+"textmining_with_structured_directory/src/data/"
    demiLen=int(len(modelfinal)/2)
    modelList1 = [modelfinal[i] for i in range(demiLen)]
    modelList2 = [modelfinal[i] for i in range((demiLen+1),len(modelfinal))]

   
    Thread1 = BestModelFinder(X=XtrainClean.description,Y=XtrainClean.Labels,ListParamModel=modelList1,path=path_to_modele)
    Thread2 = BestModelFinder(XtrainClean.description,Y=XtrainClean.Labels,ListParamModel=modelList2,path=path_to_modele)

    print("Execute and save modele:")
    Thread1.start()
    Thread2.start()

    Thread1.join()
    Thread2.join()
