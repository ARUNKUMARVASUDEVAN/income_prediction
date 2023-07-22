import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class Predictpipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            logging.info("Preprocessing started on the Prediction pipeline")
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            transformed_data=preprocessor.transform(features)
            
            pred=model.predict(transformed_data)

            return pred
        
        except Exception as e:
            logging.info('Exception occured in the prediction pipeline')
            raise CustomException(e,sys)
        
       
class CustomDataInput:
    def __init__(
                  self,_id:int,age:int,fnlwgt:int,education:int,education_num:int,capital_gain:int,
                  capital_loss:int,hours_per_week:int,workclass_Local_gov:int,
                  workclass_Never_worked:int,workclass_Private:int,workclass_Self_emp_inc:int,
                  workclass_Self_emp_not_inc:int,workclass_State_gov:int,workclass_Without_pay:int,
                  marital_status_Married_AF_spouse:int,marital_status_Married_civ_spouse:int,
                  marital_status_Married_spouse_absent:int,marital_status_Never_married:int,
                  marital_status_Separated:int,marital_status_Widowed:int,occupation_Adm_clerical:int,
                  occupation_Armed_Forces:int,occupation_Craft_repair:int,occupation_Exec_managerial:int,
                  occupation_Farming_fishing:int,occupation_Handlers_cleaners:int,occupation_Machine_op_inspct:int,
                  occupation_Other_service:int,occupation_Priv_house_serv:int,occupation_Prof_specialty:int,
                  occupation_Protective_serv:int,occupation_Sales:int,occupation_Tech_support:int,occupation_Transport_moving:int,
                  relationship_Not_in_family:int,relationship_Other_relative:int,
                  relationship_Own_child:int,relationship_Unmarried:int,relationship_Wife:int,
                  race_Asian_Pac_Islander:int,race_Black:int,race_Other:int,race_White:int,
                  sex_Male:int,sex_Female:int,country_Canada:int,country_China:int,country_Columbia:int,
                  country_Cuba:int,country_Dominican_Republic:int,country_Ecuador:int,
                  country_El_Salvador:int,country_England:int,country_France:int,
                  country_Germany:int,country_Greece:int,country_Guatemala:int,
                  country_Haiti:int,country_Holand_Netherlands:int,country_Honduras:int,
                  country_Hong:int,country_Hungary:int,country_India:int,
                  country_Iran:int,country_Ireland:int,country_Italy:int,
                  country_Jamaica:int,country_Japan:int,country_Laos:int,
                  country_Mexico:int,country_Nicaragua:int,country_Outlying_Us:0,
                  country_Peru:int,country_Philippines:int,country_Poland:int,
                  country_Portugal:int,country_Puerto_Rico:int,country_Scotland:int,
                  country_South:int,country_Taiwan:int,country_Thailand:int,
                  country_Trinadad_Tobago:int,country_United_States:int,
                  country_Vietnam:int,country_Yugoslavia:int):
        self.sex_Female=sex_Female
        self._id=_id
        self.education=education
        self.education_num = education_num
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.age = age
        self.fnlwgt = fnlwgt
        self.workclass_Local_gov = workclass_Local_gov
        self.workclass_Never_worked = workclass_Never_worked
        self.workclass_Private = workclass_Private
        self.workclass_Self_emp_inc = workclass_Self_emp_inc
        self.workclass_Self_emp_not_inc = workclass_Self_emp_not_inc
        self.workclass_State_gov = workclass_State_gov
        self.workclass_Without_pay = workclass_Without_pay
        self.marital_status_Married_AF_spouse = marital_status_Married_AF_spouse
        self.marital_status_Married_civ_spouse = marital_status_Married_civ_spouse
        self.marital_status_Married_spouse_absent = marital_status_Married_spouse_absent
        self.marital_status_Never_married = marital_status_Never_married
        self.marital_status_Separated = marital_status_Separated
        self.marital_status_Widowed = marital_status_Widowed
        self.occupation_Adm_clerical = occupation_Adm_clerical
        self.occupation_Armed_Forces = occupation_Armed_Forces
        self.occupation_Craft_repair = occupation_Craft_repair
        self.occupation_Exec_managerial = occupation_Exec_managerial
        self.occupation_Farming_fishing = occupation_Farming_fishing
        self.occupation_Handlers_cleaners = occupation_Handlers_cleaners
        self.occupation_Machine_op_inspct = occupation_Machine_op_inspct
        self.occupation_Other_service = occupation_Other_service
        self.occupation_Priv_house_serv = occupation_Priv_house_serv
        self.occupation_Prof_specialty = occupation_Prof_specialty
        self.occupation_Protective_serv = occupation_Protective_serv
        self.occupation_Sales = occupation_Sales
        self.occupation_Tech_support = occupation_Tech_support
        self.occupation_Transport_moving = occupation_Transport_moving
        self.relationship_Not_in_family = relationship_Not_in_family
        self.relationship_Other_relative = relationship_Other_relative
        self.relationship_Own_child = relationship_Own_child
        self.relationship_Unmarried = relationship_Unmarried
        self.relationship_Wife = relationship_Wife
        self.race_Asian_Pac_Islander = race_Asian_Pac_Islander
        self.race_Black = race_Black
        self.race_Other = race_Other
        self.race_White = race_White
        self.sex_Male = sex_Male
        self.country_Canada = country_Canada
        self.country_China = country_China
        self.country_Columbia = country_Columbia
        self.country_Cuba = country_Cuba
        self.country_Dominican_Republic = country_Dominican_Republic
        self.country_Ecuador = country_Ecuador
        self.country_El_Salvador = country_El_Salvador
        self.country_England = country_England
        self.country_France = country_France
        self.country_Germany = country_Germany
        self.country_Greece = country_Greece
        self.country_Guatemala = country_Guatemala
        self.country_Haiti = country_Haiti
        self.country_Holand_Netherlands = country_Holand_Netherlands
        self.country_Honduras = country_Honduras
        self.country_Hong = country_Hong
        self.country_Hungary = country_Hungary
        self.country_India = country_India
        self.country_Iran = country_Iran
        self.country_Ireland = country_Ireland
        self.country_Italy = country_Italy
        self.country_Jamaica = country_Jamaica
        self.country_Japan = country_Japan
        self.country_Laos = country_Laos
        self.country_Mexico = country_Mexico
        self.country_Nicaragua = country_Nicaragua
        self.country_Outlying_Us = country_Outlying_Us
        self.country_Peru = country_Peru
        self.country_Philippines = country_Philippines
        self.country_Poland = country_Poland
        self.country_Portugal = country_Portugal
        self.country_Puerto_Rico = country_Puerto_Rico
        self.country_Scotland = country_Scotland
        self.country_South = country_South
        self.country_Taiwan = country_Taiwan
        self.country_Thailand = country_Thailand
        self.country_Trinadad_Tobago = country_Trinadad_Tobago
        self.country_United_States = country_United_States
        self.country_Vietnam = country_Vietnam
        self.country_Yugoslavia = country_Yugoslavia
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                '_id': [self._id],
                'age': [self.age],
                'fnlwgt': [self.fnlwgt],
                'education':[self.education],
                'education_num': [self.education_num],
                'capital_gain': [self.capital_gain],
                'capital_loss': [self.capital_loss],
                'hours_per_week': [self.hours_per_week],
                'workclass_Local_gov': [self.workclass_Local_gov],
                'workclass_Never_worked': [self.workclass_Never_worked],
                'workclass_Private': [self.workclass_Private],
                'workclass_Self_emp_inc': [self.workclass_Self_emp_inc],
                'workclass_Self_emp_not_inc': [self.workclass_Self_emp_not_inc],
                'workclass_State_gov': [self.workclass_State_gov],
                'workclass_Without_pay': [self.workclass_Without_pay],
                'marital_status_Married_AF_spouse': [self.marital_status_Married_AF_spouse],
                'marital_status_Married_civ_spouse': [self.marital_status_Married_civ_spouse],
                'marital_status_Married_spouse_absent': [self.marital_status_Married_spouse_absent],
                'marital_status_Never_married': [self.marital_status_Never_married],
                'marital_status_Separated': [self.marital_status_Separated],
                'marital_status_Widowed': [self.marital_status_Widowed],
                'occupation_Adm_clerical': [self.occupation_Adm_clerical],
                'occupation_Armed_Forces': [self.occupation_Armed_Forces],
                'occupation_Craft_repair': [self.occupation_Craft_repair],
                'occupation_Exec_managerial': [self.occupation_Exec_managerial],
                'occupation_Farming_fishing': [self.occupation_Farming_fishing],
                'occupation_Handlers_cleaners': [self.occupation_Handlers_cleaners],
                'occupation_Machine_op_inspct': [self.occupation_Machine_op_inspct],
                'occupation_Other_service': [self.occupation_Other_service],
                'occupation_Priv_house_serv': [self.occupation_Priv_house_serv],
                'occupation_Prof_specialty': [self.occupation_Prof_specialty],
                'occupation_Protective_serv': [self.occupation_Protective_serv],
                'occupation_Sales': [self.occupation_Sales],
                'occupation_Tech_support': [self.occupation_Tech_support],
                'occupation_Transport_moving': [self.occupation_Transport_moving],
                'relationship_Not_in_family': [self.relationship_Not_in_family],
                'relationship_Other_relative': [self.relationship_Other_relative],
                'relationship_Own_child': [self.relationship_Own_child],
                'relationship_Unmarried': [self.relationship_Unmarried],
                'relationship_Wife': [self.relationship_Wife],
                'race_Asian_Pac_Islander': [self.race_Asian_Pac_Islander],
                'race_Black': [self.race_Black],
                'race_Other': [self.race_Other],
                'race_White': [self.race_White],
                'sex_Female':[self.sex_Female],
                'sex_Male': [self.sex_Male],
                'country_Canada': [self.country_Canada],
                'country_China': [self.country_China],
                'country_Columbia': [self.country_Columbia],
                'country_Cuba': [self.country_Cuba],
                'country_Dominican-Republic': [self.country_Dominican_Republic],
                'country_Ecuador': [self.country_Ecuador],
                'country_El_Salvador': [self.country_El_Salvador],
                'country_England': [self.country_England],
                'country_France': [self.country_France],
                'country_Germany': [self.country_Germany],
                'country_Greece': [self.country_Greece],
                'country_Guatemala': [self.country_Guatemala],
                'country_Haiti': [self.country_Haiti],
                'country_Holand-Netherlands': [self.country_Holand_Netherlands],
                'country_Honduras': [self.country_Honduras],
                'country_Hong': [self.country_Hong],
                'country_Hungary': [self.country_Hungary],
                'country_India': [self.country_India],
                'country_Iran': [self.country_Iran],
                'country_Ireland': [self.country_Ireland],
                'country_Italy': [self.country_Italy],
                'country_Jamaica': [self.country_Jamaica],
                'country_Japan': [self.country_Japan],
                'country_Laos': [self.country_Laos],
                'country_Mexico': [self.country_Mexico],
                'country_Nicaragua': [self.country_Nicaragua],
                'country_Outlying_Us': [self.country_Outlying_Us],
                'country_Peru': [self.country_Peru],
                'country_Philippines': [self.country_Philippines],
                'country_Poland': [self.country_Poland],
                'country_Portugal': [self.country_Portugal],
                'country_Puerto_Rico': [self.country_Puerto_Rico],
                'country_Scotland': [self.country_Scotland],
                'country_South': [self.country_South],
                'country_Taiwan': [self.country_Taiwan],
                'country_Thailand': [self.country_Thailand],
                'country_Trinadad&Tobago': [self.country_Trinadad_Tobago],
                'country_United-States': [self.country_United_States],
                'country_Vietnam': [self.country_Vietnam],
                'country_Yugoslavia': [self.country_Yugoslavia]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
           
        
            
        except Exception as e:
            logging.info('Exception noccured in prediction pipeliune')
            raise CustomException(e,sys)
