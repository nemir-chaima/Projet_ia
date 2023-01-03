#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:20:14 2022

@author: chaimanemir
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import hvplot.pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm
import seaborn as sns


df = pd.read_csv("/Users/chaimanemir/Desktop/projet ia/accepted_2007_to_2018Q4.csv",sep=',')
df2 =df.sample(frac = 0.45, axis = 0, random_state = 0).reset_index(drop = True)
print(df2.info())
print(df2.describe())


proportion =(pd.value_counts(df2['loan_status']))/len(df2['loan_status'])*100
pl=df2['loan_status'].value_counts().hvplot.barh(title=" loan status  counts",xlabel=" current status",ylabel='count')
hvplot.show(pl)

# -------------------- feature selection ---------------------------

# la moyenne des valeurs manquantes pour chaque colonnes 
nul = df2.isnull().mean().sort_values()
nul = nul[nul>0.3] # on supprime les colones dont le taux des valeurs manquantes depasse 0.3 (30%)
nul_col = nul.sort_values(ascending = False).index
data = df2.drop(nul_col, axis = 1)
# on supprime les valeurs manquantes sur les lignes 
data = data.dropna(axis = 0).reset_index(drop = True)


# supprimer les colonnes dont la valeurs est differentes pour chaque ligne ( ca importe aucune informations)
categorie = [feature for feature in data.columns if data[feature].dtype == "O"]
for value in categorie:
    print(value, " : ", data[value].nunique())   
data = data.drop(['id', 'url','sub_grade','title', 'zip_code','emp_title'], axis = 1)


#les colonnes a garder 

keep_list = ['charged_off','funded_amnt','addr_state', 'annual_inc', 'application_type', 
             'dti', 'earliest_cr_line', 'emp_length',  'fico_range_high', 'fico_range_low',
             'grade', 'home_ownership',  'initial_list_status', 'installment', 'int_rate', 
             'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 
             'purpose', 'revol_bal', 'revol_util',  'term','total_acc', 'verification_status','last_pymnt_amnt',
             'num_actv_rev_tl', 'mo_sin_rcnt_rev_tl_op','mo_sin_old_rev_tl_op',"bc_util","bc_open_to_buy",
             "avg_cur_bal","acc_open_past_24mths" ]
drop_list = [col for col in data.columns if col not in keep_list]
data=data.drop(labels=drop_list, axis=1)


#supprimer les colonnes qui n'ont pas une forte correlation avec le label
# correlations = data.corr()
# corr_charged_off = abs(correlations['loan_status']).sort_values(ascending=False)
# print(corr_charged_off)

#la proportion  des pret pour chaque type de status
proportion =(pd.value_counts(data['loan_status']))/len(data['loan_status'])*100

pl=data['loan_status'].value_counts().hvplot.barh(title=" loan status  counts",xlabel=" current status", ylabel='count')

hvplot.show(pl)
#------------------------- analyser les feautures and inhot coding -------------------------------------
data['grade']=data['grade'].replace(['A','B','C','D','E','F','G'],[6,5,4,3,2,1,0])
#supprimer les colonnes contenats des information apres que le pret soit approuvé
to_del=['funded_amnt','last_pymnt_amnt','addr_state']
data=data.drop(labels=to_del, axis=1)



data=data.drop(labels=['earliest_cr_line'], axis=1)

#verifier si y'a une relation entre emp_length et charged off loan_status 
emp_charged_off = data[data['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fully_paid = data[data['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
percentage_charged_off = (emp_charged_off * 100)/(emp_charged_off + emp_fully_paid)
plt.figure(figsize=(12,4), dpi=130)
percentage_charged_off.plot(kind='bar', cmap='viridis')


#on remarque qu'il n y'a pas vraiment de difference entre ceux qui travaillent pendant 
#des annees et ceux qui viennent de commencer par rapport aux payement du pret
#donc on peut supp cette colonne pcq elle n'importe pas bcp d'info pour la prediction
data= data.drop(labels=['emp_length'], axis=1)

# 
#verifier si ya une relation entre home aownership et le rayement du pret
data['home_ownership'] = data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
charged_off = data[data['loan_status']=="Charged Off"].groupby("home_ownership").count()['loan_status']
fully_paid = data[data['loan_status']=="Fully Paid"].groupby("home_ownership").count()['loan_status']
percentage_charged_off = (charged_off * 100)/(charged_off + fully_paid)
percentage_charged_off.plot(kind='bar', cmap='viridis')
plt.title("Percentage charged off per home_ownership category")

dummies_home_ownership = pd.get_dummies(data['home_ownership'])
data = pd.concat([data.drop('home_ownership', axis=1), dummies_home_ownership], axis=1)


#verifier pour leapplication type
charged_off = data[data['loan_status']=="Charged Off"].groupby("application_type").count()['loan_status']
fully_paid = data[data['loan_status']=="Fully Paid"].groupby("application_type").count()['loan_status']
percentage_charged_off = (charged_off * 100)/(charged_off + fully_paid)
percentage_charged_off.plot(kind='bar', cmap='viridis')
plt.title("Percentage charged off per application_type category")
dummies_application_type = pd.get_dummies(data['application_type'])
data = pd.concat([data.drop('application_type', axis=1), dummies_application_type], axis=1)

#pour initial_list_status 

charged_off = data[data['loan_status']=="Charged Off"].groupby("initial_list_status").count()['loan_status']
fully_paid = data[data['loan_status']=="Fully Paid"].groupby("initial_list_status").count()['loan_status']
percentage_charged_off = (charged_off * 100)/(charged_off + fully_paid)
percentage_charged_off.plot(kind='bar', cmap='viridis')
plt.title("Percentage charged off per initial_list_status category")
#il ya pas vraiment une difference entre w , f et le payement du pret donc on va la suppr
data= data.drop(labels=['initial_list_status'], axis= 1)

# pour init_rate ( taux d'interet)

sns.boxplot(data=data, y='loan_status', x='int_rate', palette='viridis')
# une grande difference entre le taux d'interet et le payement charged off / fully paid du pret a garder
#pour loan_amnt
sns.boxplot(data=data, y='loan_status', x='loan_amnt', palette='viridis')
#difference clair a garder

#num_actv_bc_tl num des carte bancaire active 
sns.boxplot(data=data, y='loan_status', x='num_actv_bc_tl', palette='viridis')
#on gardee 

#mort_acc credit immobilier
sns.boxplot(data=data, y='loan_status', x='mort_acc', palette='viridis')
#a garder aussi :
#total_acc ( nombre total des credit de pret) , total_cur_bal ( balance of the account),open_acc
#pub_rec (Number of derogatory public records), pub_rec_bankruptcies, revol_bal(Total credit revolving balance.),
#evol_util(Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.)

#purpose (raison du pret),
plt.figure(figsize=(14,6))
charged_off = data[data['loan_status']=="Charged Off"].groupby("purpose").count()['loan_status']
fully_paid = data[data['loan_status']=="Fully Paid"].groupby("purpose").count()['loan_status']
percentage_charged_off = (charged_off * 100)/(charged_off + fully_paid)
percentage_charged_off.plot(kind='bar', cmap='viridis')
plt.title("Percentage charged off per purpose category")
dummies_purpose = pd.get_dummies(data['purpose'])
data= pd.concat([data.drop('purpose', axis=1), dummies_purpose], axis=1)

#term
data['term'] = data['term'].apply(lambda x: int(x[0:3]))
sns.countplot(data=data, x='term', palette='viridis')
charged_off = data[data['loan_status']=="Charged Off"].groupby("term").count()['loan_status']
fully_paid = data[data['loan_status']=="Fully Paid"].groupby("term").count()['loan_status']
percentage_charged_off = (charged_off * 100)/(charged_off + fully_paid)
percentage_charged_off.plot(kind='bar', cmap='viridis')
plt.title("Percentage charged off per term category")
dummies_term = pd.get_dummies(data['term'])
data=pd.concat([data.drop('term', axis=1), dummies_term], axis=1)

#verification_status (Indicates if income was verified by LC, not verified, or if the income source was verified.)
charged_off = data[data['loan_status']=="Charged Off"].groupby("verification_status").count()['loan_status']
fully_paid = data[data['loan_status']=="Fully Paid"].groupby("verification_status").count()['loan_status']
percentage_charged_off = (charged_off * 100)/(charged_off + fully_paid)
percentage_charged_off.plot(kind='bar', cmap='viridis')
plt.title("Percentage charged off per verification_status category")
dummies_verification_status = pd.get_dummies(data['verification_status'])
data = pd.concat([data.drop('verification_status', axis=1), dummies_verification_status], axis=1)


#------------------------ inhot coding  of target ---------------------------
#situation A = pret sera remboursé : fully paid , current
#situation B = pret ne sera pas remboursé : charged off 
#situation C = on peut rien conclure  : late 1 , late 2 , in grace

a_supp_index= data[data['loan_status']=='Late (16-30 days)'].index
data = data.drop(labels=a_supp_index).reset_index(drop=True)
a_supp_index= data[data['loan_status']=='Late (31-120 days)'].index
data = data.drop(labels=a_supp_index).reset_index(drop=True)
a_supp_index= data[data['loan_status']=='In Grace Period'].index
data = data.drop(labels=a_supp_index).reset_index(drop=True)

data['target']=0
data['target'][(data['loan_status']=='Fully Paid')]= 1
data['target'][(data['loan_status']=='Current')]= 1
data=data.drop(labels=['loan_status'], axis=1)

#------------------------ split  ----------------------------
X=data.drop(labels=['target'], axis=1)
Y=  data['target']
# 
X_train,X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.15)

#---------------------------scalling ------------------------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#------------------------- Model --------------------------------

#XGBoost

model = XGBClassifier()
model.fit(X_train, Y_train)
preds = model.predict(X_test) 
accuracy_report_XGBoost=classification_report(Y_test,preds)
print(accuracy_report_XGBoost)
plot_confusion_matrix(model,X_test,Y_test)
# score_xgb=model.score(X_test,Y_test)

#random_forest

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,Y_train)
preds = rf.predict(X_test)
accuracy_report_RandomForest = classification_report(Y_test,preds)
print(accuracy_report_RandomForest)
plot_confusion_matrix(rf,X_test,Y_test)
# score_rf=rf.score(X_test,Y_test)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
lr =LogisticRegression()
lr.fit(X_train,Y_train)
preds = lr.predict(X_test)
accuracy_report_LinReg = classification_report(Y_test,preds)
print(accuracy_report_LinReg)
plot_confusion_matrix(lr,X_test,Y_test)
# score_lr=lr.score(X_test,Y_test)
#--------- Reseaux de neurones -----------------
#ANN

model = keras.Sequential()
# input couche
model.add(Dense(119,  activation='relu'))
model.add(Dropout(0.1))

#  couches cachées
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.1))

# output couche
model.add(Dense(units=1,activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=Y_train, epochs=10,batch_size=256,validation_data=(X_test, Y_test), )

losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
plt.show()

predictions = (model.predict(X_test) > 0.5).astype("int32")
accuracy_report_ANN=classification_report(Y_test,predictions)
print(accuracy_report_ANN)

cm = confusion_matrix(Y_test,predictions)
f = sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', square=True)




predictions = (model.predict(X_test) > 0.7).astype("int32")
accuracy_report_ANN=classification_report(Y_test,predictions)
print(accuracy_report_ANN)

cm = confusion_matrix(Y_test,predictions)
f = sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', square=True)

