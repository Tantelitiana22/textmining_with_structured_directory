from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

class ModelDeepLearning:
    def __init__(self,choice_model=None):
        self.choice_model=choice_model
        self.model = None
        self.uniqueLabel =None
        self.tokenizer = Tokenizer()
        self.maxlen = None

    def fit(self,X,Y):
        
        tab=[]
        for i,k in enumerate(X):
            tab.append(len(k.split(" ")))
        self.maxlen = np.max(tab)
        self.uniqueLabel =np.unique(Y)
        self.tokenizer.fit_on_texts(X)
        X_train_keras = self.tokenizer.texts_to_sequences(X)
        
        X_train_keras1 = pad_sequences(X_train_keras, padding='post', maxlen=self.maxlen)
        
        le =LabelEncoder()
        le.fit(trainData.Labels)
        Ya=to_categorical(le.transform(Y))
        embedding_dim = 50
        
        if self.choice_model =="CNN":
            vocab_size=len(self.tokenizer.word_index)+1
            self.model = Sequential()
            self.model.add(layers.Embedding(vocab_size, embedding_dim, input_length=self.maxlen))
            self.model.add(layers.Conv1D(100, 10, activation='relu'))
            self.model.add(layers.GlobalMaxPooling1D())
            self.model.add(layers.Dense(10, activation='relu'))
            self.model.add(layers.Dense(11, activation='softmax'))
            self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            history = self.model.fit(X_train_keras1, Ya,epochs=10,verbose=False,validation_split=0.25,batch_size=10)
            
            return self
            
        elif self.choice_model == "LSTM":
            vocab_size=len(self.tokenizer.word_index)+1
            self.model = Sequential()
            self.model.add(layers.Embedding(vocab_size, embedding_dim, input_length=self.maxlen))
            self.model.add(layers.Dense(50, activation='relu'))
            self.model.add(layers.LSTM(200, dropout=0.2, recurrent_dropout=0.2))
            self.model.add(layers.Dense(11, activation='softmax'))
            self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            history = self.model.fit(X_train_keras1, Ya,epochs=10,verbose=False,validation_split=0.25,batch_size=10)
                
            return self
        
        else:
        
            vocab_size=len(self.tokenizer.word_index)+1
            self.model = Sequential()
            self.model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,input_length=self.maxlen))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(30, activation='relu'))
            self.model.add(layers.Dense(15, activation='relu'))
            self.model.add(layers.Dense(11, activation='softmax'))

            self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
            history = self.model.fit(X_train_keras1, Ya,epochs=10,verbose=False,validation_split=0.25,batch_size=10)
         
            return self
        
    def predict(self,X,Y=None):
        

        X_test_keras = self.tokenizer.texts_to_sequences(X)
        X_test_keras1 = pad_sequences(X_test_keras, padding='post', maxlen=self.maxlen)
        
        pred = self.model.predict_classes(X_test_keras1)
        predLabel = [self.uniqueLabel[k] for k in pred]
        
        return predLabel
    
# if __name__=="__main__":
#     model1=ModelDeepLearning()
#     model2=ModelDeepLearning("CNN")
#     model3=ModelDeepLearning("LSTM")
#     model1.fit(trainData.resume,trainData.Labels)
#     pickle.dump(model1,open("../data/model_with_resum/sequence_simpleneuralnetwork.sav","wb"))   
#     model2.fit(trainData.resume,trainData.Labels)
#     pickle.dump(model2,open("../data/model_with_resum/sequence_CNNmodel.sav","wb"))    
#     model3.fit(trainData.resume,trainData.Labels)
#     pickle.dump(history,open("../data/model_with_resum/sequence_LMST.sav","wb"))
