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

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=10)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an instance of the data_transformation class and perform data transformation
    data_transformer =DataTransformation()
    train_arr, test_arr,preprocessor_obj_path = data_transformer.initiate_data_transformation(train_data, test_data)
