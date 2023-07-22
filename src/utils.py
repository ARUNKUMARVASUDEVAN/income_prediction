import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as pt
sns.set(color_codes=True)
import pymongo
import csv
from imblearn.over_sampling import RandomOverSampler
import pickle
import os
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,roc_auc_score

class Mongo_db_data:
    def __init__(self, db_name, collection_name):
        self.db_name = db_name
        self.collection_name = collection_name

    def fetch(self):
        logging.info('fetching has started')
        try:
            client = pymongo.MongoClient("mongodb://localhost:27017/")
            db = client[self.db_name]
            collection = db[self.collection_name]
            cursor = collection.find()
            data_list = []
            # Iterate through the cursor and append each document to the list
            for doc in cursor:
                data_list.append(doc)

            data = pd.DataFrame(data_list)

            return data
        except Exception as e:
            logging.info("Yo Error occurred on Mongo_db_data")
            raise CustomException(e, sys)

def transform(data):
        x = data.drop('salary', axis=1)
        y = data['salary']
        random_sampling = RandomOverSampler()
        x_sampled, y_sampled = random_sampling.fit_resample(x, y)

        # Combine the features and target into a single DataFrame
        balanced_data = x_sampled.copy()
        balanced_data['salary'] = y_sampled

        return balanced_data

def save_model(file_path,obj):
     try:
          dir_path=os.path.dirname(file_path)
          os.makedirs(dir_path,exist_ok=True)
          with open(file_path,"wb") as file_obj:
               pickle.dump(obj,file_obj)
     except Exception as e:
          raise CustomException(e,sys)
     
    
def Model_evaluater(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        
        for model_name,model in models.items():
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            
            
            accuracy=accuracy_score(y_test,y_pred)*100
            confusion=confusion_matrix(y_test,y_pred)
            precision=precision_score(y_test,y_pred,average='weighted')*100
            roc=roc_auc_score(y_test,y_pred)*100
            f1=f1_score(y_test,y_pred)*100
            report[model_name]=accuracy,confusion,precision,roc,f1

        return report
    except Exception as e:
        logging.info('Exception occured during model Training')
        raise CustomException(e,sys)


def load_object(file_path):
     try:
          with open(file_path,'rb') as file_obj:
               return pickle.load(file_obj)
     except Exception as e:
          logging.info('Exception occured in load_object function utils')
          raise CustomException(e,sys)