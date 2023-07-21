import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.utils import Model_evaluater









if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Create an instance of the data_transformation class and perform data transformation
    data_transformer =DataTransformation()
    train_arr, test_arr,preprocessor_obj_path = data_transformer.initiate_data_transformation(train_data_path,test_data_path)
    