# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:28:03 2022

@author: JELENA
"""

# %% Import tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Import and check data

data = pd.read_csv('heart.csv')
columns = data.columns
types = data.dtypes

''' among attributes there are 5 categorical ones
in this case we assume those are nominal ones, because none of the categorical attributes have order among it '''

''' the best way to deal with those variables in this case is one-hot encoding
one-hot encoding is usually followed with the dimensionality enlargment
good practice is to use one-hot encoding followed by pca for dimmensionality reduction
one-hot encoding equals dummy variables
both will be tested here
first categorical attribute is 'Sex' - this will not be encoded, but just replaced 0 for males and 1 for females
second one is chest pain type, and we have 4 different chest pain types - we will use one hod encoding
third is resting ecg with 3 types
fourth is excersive angina, with also binary options
fifth is st slope with 3 types '''


#%% one-hot (dummy) encoding

data = pd.get_dummies(data)

''' since we have 2 categorical variables, with 2 possible options, we need to drop 2 columns made by dummy vectors
one column is sex for male/female, other is exercise angina'''

data = data.drop(columns = ['Sex_F', 'ExerciseAngina_Y'])
data = (data-data.min())/(data.max()-data.min())

target = data['HeartDisease']
data = data.drop(columns = ['HeartDisease'])


#%% Logistic regression classification

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
print('Accuracy is: ', accuracy_score(y_test,y_pred))

print('Classification report:')
print(classification_report(y_test,y_pred))

print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True, cbar = False)

print('Area under roc curve is: ', roc_auc_score(y_test,y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
f = plt.figure()
plt.plot(fpr,tpr)
plt.title('ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

#%% Logistic regression classification with cross validation


''' there is not hige difference between number of people with heart disease and without heart disease
k fold cross validation could be used, but stratified k fold cross validation because we need to 
make surethere is equal number of people from both categories in each split '''


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
n = 5
kfold = StratifiedKFold(n_splits = n, shuffle = True)

acc = 0
auc = 0
cm = 0
tpr = 0 #true positive rate/sensitivity
for train_index, test_index in kfold.split(data, target):

    #print("Train indices:", train_index, "Test indices:", test_index)
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    print('Accuracy is: ', accuracy_score(y_test,y_pred))
    
    print('Area under roc curve is: ', roc_auc_score(y_test,y_pred))
        
    print('Confusion matrix:')
    print(confusion_matrix(y_test,y_pred))
    
    print('Classification report:')
    print(classification_report(y_test,y_pred))
     
    acc = acc + accuracy_score(y_test,y_pred)
    auc = auc + roc_auc_score(y_test,y_pred)
    cm = cm + confusion_matrix(y_test,y_pred)
    tpr = cm[1,1]/(cm[1,1]+cm[0,1])
    

average_model_acc = acc/n * 100
average_model_auc = auc/n
average_model_cm = cm/n


f = plt.figure()
plt.bar(0, average_model_acc, color = 'r')
plt.bar(1, average_model_auc, color = 'b')
plt.xticks([0, 1], ["accuracy", "area under roc curve"])
plt.ylim(np.mean([average_model_acc, average_model_auc])/2, 1)
    
    
f = plt.figure()
plt.title('Confusion matrix')
sns.heatmap(average_model_cm, annot = True, cbar = False)



''' Since we are dealing here with health tech problem, accuracy is not the ideal metric we want to track.
We need to make sure we have high rate for true positives and false negatives.
Because in health related problem the cost of mistake is high. '''


''' Model made in this code has 0.86 value for area under roc curve, and accuracy of 86%
These metrics are quite good, but not satisfying for purpose of the model. '''

''' Further improvements of the model can go in multiple ways:
    1. gather more data - 918 samples of data is really low in comparisson to number of heart
    diseases that happen each year.
    2. test this dataset with different machine learning algorithms.
    3. gather better attributes: for example cholesterol attribute in this dataset presents the total blood cholesterol.
    That attribute could be separated in 3 different ones (HDL, LDL and tryglicerides), and each of those could have different impact
    on heart disease
    4. colaborate with professionals(doctors) to learn more about the specific problem. '''