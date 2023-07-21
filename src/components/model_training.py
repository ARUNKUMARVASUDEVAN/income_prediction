import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,auc

from src.utils import save_model
from src.utils import Model_evaluater
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_data_input,test_data_input,train_data_target,test_data_target):
        try:
            train_data_input=train_data_input.fillna(method='ffill')
            train_data_target=train_data_target.fillna(method='ffill')
            test_data_target=test_data_target.fillna(method='ffill')
            test_data_input=test_data_input.fillna(method='ffill')
            logging.info('Splitting Dependent and Independent Variables from train and test data')
            X_train,y_train,X_test,y_test=train_data_input,train_data_target,test_data_input,test_data_target
            
            
            models={
            "Logistic_regression":LogisticRegression(max_iter=1000),
            "Decision_tree":DecisionTreeClassifier(),
            "SVM":SVC(),
            "Random_forest_classifier":RandomForestClassifier(),
            "Naive_bayes":BernoulliNB()
            }
            model_report:dict=Model_evaluater(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n==========================================================\n')
            logging.info(f'Model Report :{model_report}')
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            print(f'Best Model Found, Model Name :{best_model_name},accuracies:{best_model_score}')
            print('\n============================================================\n')
            logging.info(f'Best Model Found,Model Name :{best_model_name},accuracies :{best_model_score}')

            save_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)



