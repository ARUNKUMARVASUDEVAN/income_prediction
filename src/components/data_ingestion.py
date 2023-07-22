import pandas as pd
import pymongo
import csv
from src.logger import logging
from src.exception import CustomException
from src.utils import Mongo_db_data
from src.utils import transform
from src.components.data_transformation import DataTransformation
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts')
        try:
            data = Mongo_db_data('Internship', 'Adult')
            logging.info("Dataset read from MongoDB")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df = data.fetch()
            df = transform(df)
            df=pd.DataFrame(df)

            
            X=df.drop('salary',axis=1)
            y=df['salary']
            
            # Combine the features and target into a single DataFrame
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train test split')
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)

            

            logging.info(f'X_train Dataframe Head:\n{X_train.head().to_string()}')

            X_test.loc[200,'workclass']='Never-worked'
            X_test.loc[200,'country'] ='Holand-Netherlands'
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            y_train=y_train.replace({' <=50K':0,' >50K':1})
            y_test =y_test.replace({' <=50K': 0, ' >50K': 1})
            
            

            # Concatenate along the columns (axis=1) with train_set_X and train_y as a list
            train_set = pd.concat([X_train,y_train], axis=1)
            test_set = pd.concat([X_test,y_test], axis=1)
            

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            # Return train_set and test_set instead of paths
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()

    # Unpack the returned values correctly
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformer =DataTransformation()
    input_column_train_arr,input_column_test_arr,target_column_train_df,target_column_test_df,preprocessor_obj_path = data_transformer.initiate_data_transformation('D:/internship/income/artifacts/train.csv','D:/internship/income/artifacts/test.csv')
    