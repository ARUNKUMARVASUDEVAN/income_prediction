
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException 
from src.logger import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.utils import save_model
from sklearn.base import BaseEstimator,TransformerMixin
from imblearn.over_sampling import RandomOverSampler

class Imputer(BaseEstimator,TransformerMixin):
    def fit(self,df):
        return self
    def transform(self,df):
        df['age']=df['age'].fillna(df['age'].median())
        df['workclass']=df['workclass'].fillna(df['workclass'].mode()[0])
        df['marital-status']=df['marital-status'].fillna(df['marital-status'].mode()[0])
        df['occupation']=df['occupation'].fillna(df['occupation'].mode()[0])
        df['race']=df['race'].fillna(df['race'].mode()[0])
        df['relationship']=df['relationship'].fillna(df['relationship'].mode()[0])
        df['sex']=df['sex'].fillna(df['sex'].mode()[0])

        return df

        

class drop_column(BaseEstimator,TransformerMixin):
    def __init__(self,columns):
        self.columns=columns
    def fit(self,data):
        return self
    def transform(self,data):
        for i in self.columns:
            data.drop(i,axis=1,inplace=True)
        return data
    
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


    

columns_to_drop=['_id','education']


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated")

            pipeline=Pipeline([
                ('Remove_spaces',remove_spaces(self)),
                ('drop_columns',drop_column(columns_to_drop)),
                ('Imputer',Imputer()),
                               
                
                ('One_hot',onehot_encoder(columns=['workclass','marital-status', 'occupation',
                'relationship', 'race', 'sex', 'country'])),
                ('FeatureScaling',FeatureScaling(columns=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']))
                
            ])
            
            return pipeline
        
        except Exception as e:
            logging.info("Exeception occured in data_transformation")
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data Transformation has started")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Read train and test data')
            logging.info(f"Train DataFrame head:\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head:\n{test_df.head().to_string()}")
            logging.info("Starting of pipeline")
            preprocessor_obj=self.get_data_transformation_object()
            target_column='salary'
            drop_column=[target_column]
            logging.info('Splitting train data as input and target')
            input_column_train_df=train_df.drop(columns=drop_column,axis=1)
            target_column_train_df=train_df[target_column]

            input_column_test_df=test_df.drop(columns=drop_column,axis=1)
            target_column_test_df=test_df[target_column]
            
            logging.info("Fitting on the Pipeline")

            input_column_train_arr=preprocessor_obj.fit_transform(input_column_train_df)
            input_column_test_arr=preprocessor_obj.transform(input_column_test_df)
            

            logging.info("Pipeline Fitting is completed")

            #train_arr=np.c_[input_column_train_arr,np.array(target_column_train_df)]
            #test_arr=np.c_[input_column_test_arr,np.array(target_column_test_df)]

            save_model(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )

            logging.info("Data Transformed")

            return(
                input_column_train_arr,
                input_column_test_arr,
                target_column_train_df,
                target_column_test_df,
                self.data_transformation_config.preprocessor_ob_file_path
            )
            
        except Exception as e:
            logging.info("Error occured in initiate data Transformaiton")
            raise CustomException(e,sys)
