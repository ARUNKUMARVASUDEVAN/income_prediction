
# 📈 Adult Census Income Prediction 

The objective of this project is to develop a predictive model that can accurately estimate the income level of adults based on various demographic, educational, and occupational features. The model should analyze a given individual's characteristics and provide a prediction of whether their income exceeds a certain threshold, such as $50,000 per year.

## Description
age: continuous.

workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.

fnlwgt: continuous.

education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

education-num: continuous.

marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.


sex: Female, Male.

capital-gain: continuous.

capital-loss: continuous.

hours-per-week: continuous.

native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Requirements
This project requires Python 3.x and the following Python libraries installed:

NumPy
Pandas
matplotlib
scikit-learn


## Classification Models Used:
🤖 Decision Tree

🤖 Logistic Regression

🤖 Random Forests

🤖 k-Nearest Neighbours

🤖 Support Vector Machine

Creating  A virtual Environment

         conda create -p venv python=3.8


## Result
After cleaning and preparing the data for the models, cross-validation has been done and other operations to compare between the models.

Finally, The Random Forest Classifier has been chosen, for achieving very high results in terms of evaluation metrics. A 93% accuracy score, also the same result for F1-score.


For Further Details Check Report On the Documents Folder
## Project Demo:https://youtu.be/ptocRTuFIhY
