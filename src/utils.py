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
     
class drop_column(BaseEstimator,TransformerMixin):
    def __init__(self,columns):
        self.columns=columns
    def fit(self,data):
        return self
    def transform(self,data):
        for i in self.columns:
            data.drop(i,axis=1,inplace=True)
        return data

class remove_spaces(BaseEstimator,TransformerMixin):
    def __init__(self,data):
        self.data=data

    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        data=X.copy()
        data.columns=data.columns.str.strip()
        data=data.apply(lambda x: x.str.strip() if x.dtype =='object' else x)
        mode=data['workclass'].mode().values[0]
        mode1=data['country'].mode().values[0]
        data['workclass']=data['workclass'].replace('?',mode)
        data['country']=data['country'].replace('?',mode1)

        return data
    
class onehot_encoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, data):
        return self
    
    def transform(self, data):
        data_encode = data.copy()
        for col in self.columns:
            enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)  # Correct handle_unknown and sparse parameters
            encoded_data = enc.fit_transform(data[[col]])  # Use fit_transform with a DataFrame
            feature_names = enc.get_feature_names_out([col])  # Correct method name and pass column name as a list
            data_encode.drop(col, axis=1, inplace=True)  # Drop the original column
            data_encode[feature_names] = encoded_data  # Add the one-hot encoded columns
        return data_encode

class FeatureScaling(BaseEstimator,TransformerMixin):
    def __init__(self,columns):
        self.columns=columns
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X_scaled=X.copy()
        scaler=StandardScaler()
        X_scaled[self.columns]=scaler.fit_transform(X[self.columns])
        return X_scaled