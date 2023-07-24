import streamlit as st
from src.pipeline.prediction_pipeline import CustomDataInput, Predictpipeline
import pickle
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
st.set_page_config(page_title='Adult Income Prediction',page_icon=':family:',layout='wide')
st.header('Adult Income Prediction')
age = st.text_input('Enter Your Age')
sex = st.selectbox('Enter Your Sex', options=['Male', 'Female'])
#fnlwgt = st.number_input('Enter Your Fnl Weight')
education_num = st.slider('Give Us Your Qualification', min_value=0, max_value=16, step=1)
race = st.selectbox("Enter Your Race", options=['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
capital_loss = st.number_input("Enter Your Capital Loss")
capital_gain = st.number_input("Enter Your Capital Gain")
relationship = st.selectbox("Select Your Relationship Status", options=['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other'])
hours_per_week = st.text_input("How Many Hours You Will Work per Week")
workclass = st.selectbox("Which Sector Are You:", options=['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
marital_status = st.selectbox("Enter Your Marital Status", options=['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
occupation = st.selectbox('Enter Your Occupation', options=['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
country = st.selectbox('Enter Your Country', options=['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'])

_id = 0
education = 0     
workclass_Local_gov= 1 if workclass == "Local-gov" else 0
workclass_Never_worked= 1 if workclass == 'Never-worked' else 0
workclass_Private= 1 if workclass == 'Private' else 0
workclass_Self_emp_inc= 1 if workclass == 'Self-emp-inc' else 0
workclass_Self_emp_not_inc= 1 if workclass == "Self-emp-not-inc" else 0
workclass_State_gov= 1 if workclass == "State-gov" else 0
workclass_Without_pay= 1 if workclass == "Without-pay" else 0
marital_status_Married_AF_spouse= 1 if marital_status == "Married-AF-spouse" else 0
marital_status_Married_civ_spouse= 1 if marital_status == "Married-civ-spouse" else 0
marital_status_Married_spouse_absent= 1 if marital_status == "Married-spouse-absent" else 0
marital_status_Never_married= 1 if marital_status == "Never-married" else 0
marital_status_Separated= 1 if marital_status == "Separated" else 0
marital_status_Widowed= 1 if marital_status == "Widowed" else 0
occupation_Adm_clerical= 1 if occupation == "Adm-clerical" else 0
occupation_Armed_Forces= 1 if occupation == "Armed-Forces" else 0
occupation_Craft_repair= 1 if occupation == "Craft-repair" else 0
occupation_Exec_managerial= 1 if occupation == "Exec-managerial" else 0
occupation_Farming_fishing= 1 if occupation == "Farming-fishing" else 0
occupation_Handlers_cleaners= 1 if occupation == "Handlers-cleaners" else 0
occupation_Machine_op_inspct= 1 if occupation == "Machine-op-inspct" else 0
occupation_Other_service= 1 if occupation == "Other-service" else 0
occupation_Priv_house_serv= 1 if occupation == "Priv-house-serv" else 0
occupation_Prof_specialty= 1 if occupation == "Prof-specialty" else 0
occupation_Protective_serv= 1 if occupation == "Protective-serv" else 0
occupation_Sales= 1 if occupation == "Sales" else 0
occupation_Tech_support= 1 if occupation == "Tech-support" else 0
occupation_Transport_moving= 1 if occupation == "Transport-moving" else 0
relationship_Not_in_family= 1 if relationship == "Not-in-family" else 0
relationship_Other_relative= 1 if relationship == "Other-relative" else 0
relationship_Own_child= 1 if relationship == "Own-child" else 0
relationship_Unmarried= 1 if relationship == "Unmarried" else 0
relationship_Wife= 1 if relationship == "Wife" else 0
race_Asian_Pac_Islander=1 if race == "Asian-Pac-Islander" else 0
race_Black= 1 if race == "Black" else 0
race_Other= 1 if race == "Other" else 0
race_White= 1 if race == "White" else 0
sex_Female= 1 if sex == "Female" else 0
sex_Male= 1 if sex == "Male" else 0
country_Canada= 1 if country == "Canada" else 0
country_China= 1 if country == "China" else 0
country_Columbia= 1 if country == "Columbia" else 0
country_Cuba= 1 if country == "Cuba" else 0
country_Dominican_Republic= 1 if country == "Dominican-Republic" else 0
country_Ecuador= 1 if country == "Ecuador" else 0
country_El_Salvador= 1 if country == "El-Salvador" else 0
country_England= 1 if country == "England" else 0
country_France= 1 if country == "France" else 0
country_Germany= 1 if country == "Germany" else 0
country_Greece= 1 if country == "Greece" else 0
country_Guatemala= 1 if country == "Guatemala" else 0
country_Haiti= 1 if country == "Haiti" else 0
country_Holand_Netherlands= 1 if country == "Holand-Netherlands" else 0
country_Honduras= 1 if country == "Honduras" else 0
country_Hong= 1 if country == "Hong" else 0
country_Hungary= 1 if country == "Hungary" else 0
country_India= 1 if country == "India" else 0
country_Iran= 1 if country == "Iran" else 0
country_Ireland= 1 if country == "Ireland" else 0
country_Italy= 1 if country == "Italy" else 0
country_Jamaica= 1 if country == "Jamaica" else 0
country_Japan= 1 if country == "Japan" else 0
country_Laos= 1 if country == "Laos" else 0
country_Mexico= 1 if country == "Mexico" else 0
country_Nicaragua= 1 if country == "Nicaragua" else 0
country_Outlying_Us=1 if country == "Outlying-US(Guam-USVI-etc)" else 0
country_Peru= 1 if country == "Peru" else 0
country_Philippines= 1 if country == "Philippines" else 0
country_Poland= 1 if country == "Poland" else 0
country_Portugal= 1 if country == "Portugal" else 0
country_Puerto_Rico= 1 if country == "Puerto-Rico" else 0
country_Scotland= 1 if country == "Scotland" else 0
country_South= 1 if country == "South" else 0
country_Taiwan= 1 if country == "Taiwan" else 0
country_Thailand= 1 if country == "Thailand" else 0
country_Trinadad_Tobago= 1 if country == "Trinadad&Tobago" else 0
country_United_States= 1 if country == "United-States" else 0
country_Vietnam= 1 if country == "Vietnam" else 0
country_Yugoslavia= 1 if country == "Yugoslavia" else 0
    

button = st.button("Check Your Income")
if button:
    model_path=os.path.join('artifacts','model.pkl')
    model=load_object(model_path)
    pred=model.predict([[age,education_num,capital_gain,capital_loss,hours_per_week,workclass_Local_gov,workclass_Never_worked,workclass_Private,workclass_Self_emp_inc,workclass_Self_emp_not_inc,workclass_State_gov,workclass_Without_pay,marital_status_Married_AF_spouse,marital_status_Married_civ_spouse,marital_status_Married_spouse_absent,marital_status_Never_married,marital_status_Separated,marital_status_Widowed,occupation_Adm_clerical,occupation_Armed_Forces,occupation_Craft_repair,occupation_Exec_managerial,occupation_Farming_fishing,occupation_Handlers_cleaners,occupation_Machine_op_inspct,occupation_Other_service,occupation_Priv_house_serv,occupation_Prof_specialty,occupation_Protective_serv,occupation_Sales,occupation_Tech_support,occupation_Transport_moving,relationship_Not_in_family,relationship_Other_relative,relationship_Own_child,relationship_Unmarried,relationship_Wife,race_Asian_Pac_Islander,race_Black,race_Other,race_White,sex_Male,sex_Female,country_Canada,country_China,country_Columbia,country_Cuba,country_Dominican_Republic,country_Ecuador,country_El_Salvador,country_England,country_France,country_Germany,country_Greece,country_Guatemala,country_Haiti,country_Holand_Netherlands,country_Honduras,country_Hong,country_Hungary,country_India,country_Iran,country_Ireland,country_Italy,country_Jamaica,country_Japan,country_Laos,country_Mexico,country_Nicaragua,country_Outlying_Us,country_Peru,country_Philippines,country_Poland,country_Portugal,country_Puerto_Rico,country_Scotland,country_South,country_Taiwan,country_Thailand,country_Trinadad_Tobago,country_United_States,country_Vietnam,country_Yugoslavia]])
    if pred==1:
        st.write('Ur Income will be >50k')
    else :
        st.write('Ur Income is =<50k')
