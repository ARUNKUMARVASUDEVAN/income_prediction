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
from src.utils import save_model
from src.utils import drop_column
from src.utils import remove_spaces
from src.utils import onehot_encoder
from src.utils import FeatureScaling

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation starts")
            drop_columns = ['_id', 'education']
            logging.info("Pipeline has started")
           
            pipeline = Pipeline([
                ('drop_columns', drop_column(drop_columns)),
                ('Remove_spaces', remove_spaces(self)),
                ('One_hot', onehot_encoder(columns=['workclass', 'marital-status', 'occupation',
                                                    'relationship', 'race', 'sex', 'country'])),
                ('FeatureScaling', FeatureScaling(columns=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']))
            ])

            return pipeline
        except Exception as e:
            logging.info("Error in the Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head :\n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_obj()
            target = 'salary'
            drop_col = ['salary']
            
            input_feature_train_df = train_df.drop(columns=drop_col, axis=1)
            target_feature_train_df = train_df[target]

            input_feature_test_df = test_df.drop(columns=drop_col, axis=1)
            target_feature_test_df = test_df[target]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_model(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataTransformation()
    train_data, test_data, preprocessor_obj_path = obj.initiate_data_transformation(train_path, test_path)

    # Here, you can use the transformed data and preprocessor_obj_path for further processing.
