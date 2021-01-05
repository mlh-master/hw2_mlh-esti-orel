import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import scipy.stats as stats
import random
from pathlib import Path
import sys
import matplotlib as mpl
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import hinge_loss
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


#Question 1

file = Path.cwd().joinpath('HW2_data.csv')
Diab = pd.read_csv(file)

def nan2value_random(df):
    #input: DataFrame
    #output: DataFrame without nans
    df = df.dropna(axis=0, thresh=15) #Of the 565 patients, we want to remove patients with at least 3 NaN.
    df_nan = df.copy()
    # Patients with 2 nan values or less: we will have the NaNs replaced with random values from a feature values (column).
    # Since this is a small number of patients from the group examined, if a bias is created it is very small, and yet the given data is large enough.
    for col in df_nan:
        bank_for_col = df_nan[col]
        bank_for_col = bank_for_col.dropna()
        bank_for_col = np.random.choice(bank_for_col, size=len(df_nan[col]))
        df_nan[col] = df_nan[col].fillna(pd.Series(bank_for_col))
    return df_nan
clean_Diab = nan2value_random(Diab)

#Question 2
X = clean_Diab[['Age','Gender','Increased Urination','Increased Thirst','Sudden Weight Loss','Weakness','Increased Hunger','Genital Thrush','Visual Blurring','Itching','Irritability','Delayed Healing','Partial Paresis','Muscle Stiffness','Hair Loss','Obesity','Family History']]
y= clean_Diab[['Diagnosis']]
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10, stratify=y)


#Question 3

# 3a
#create binary train, test DataFrames (except for age feature)
X_train_binary = X_train.replace(['Yes','Female','Positive'],value = 1)
X_train_binary = X_train_binary.replace(['No','Male','Negative'],value = 0)
x_test_binary = x_test.replace(['Yes','Female','Positive'],value = 1)
x_test_binary = x_test_binary.replace(['No','Male','Negative'],value = 0)
#create a dictionary with features and values as %train, %test, %delta
list_train = [None]*16
list_test = [None]*16
delta = [None]*16
features_dictionary={}
features_dictionary['Positive Feature']=['%Train', '%Test', '%Delta']
for i in range(0,16):
    list_train[i] = X_train_binary.iloc[:,i+1].sum()*(100/len(X_train_binary))
    list_test[i] = x_test_binary.iloc[:,i+1].sum()*(100/len(x_test_binary))
    delta[i] = abs(list_train[i]-list_test[i])
    features_dictionary[clean_Diab.columns[i+1]]= [list_train[i], list_test[i], delta[i]]

df_features_dictionary = pd.DataFrame.from_dict(features_dictionary).T
print(df_features_dictionary)

#3b
#create a dictionary that shows the relationship between feature and label.
#keys: 1.positive feature 2.negative feature
#values for each key: 1. number of positive diagnosis 2.number of negative diagnosis
#We create a bar plot after defining a dictionary for a feature
# label_feature_relationship={}
# label_feature_relationship['Female'] = {}
# label_feature_relationship['Female']['Positive'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Female')) & (clean_Diab.Diagnosis.str.contains('Positive'))])
# label_feature_relationship['Female']['Negative'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Female')) & (clean_Diab.Diagnosis.str.contains('Negative'))])
#
# label_feature_relationship['Male'] = {}
# label_feature_relationship['Male']['Positive'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Male')) & (clean_Diab.Diagnosis.str.contains('Positive'))])
# label_feature_relationship['Male']['Negative'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Male')) & (clean_Diab.Diagnosis.str.contains('Negative'))])
#
# df = pd.DataFrame.from_dict(label_feature_relationship)
# df = df.T
# df.plot.bar(rot=0, title='Gender')
# plt.ylabel('Counts')
# plt.show()
#
# for i in range(2,16):
#     title = clean_Diab.columns[i]
#     label_feature_relationship = {}
#     label_feature_relationship["Has %s" %title] = {}
#     label_feature_relationship["Has %s" %title]['Positive'] = len(clean_Diab[(clean_Diab[title].str.contains('Yes')) & clean_Diab.Diagnosis.str.contains('Positive')])
#     label_feature_relationship["Has %s" %title]['Negative'] = len(clean_Diab[(clean_Diab[title].str.contains('Yes')) & clean_Diab.Diagnosis.str.contains('Negative')])
#     label_feature_relationship["No %s" % title] = {}
#     label_feature_relationship["No %s" % title]['Positive'] = len(clean_Diab[(clean_Diab[title].str.contains('No')) & clean_Diab.Diagnosis.str.contains('Positive')])
#     label_feature_relationship["No %s" % title]['Negative'] = len(clean_Diab[(clean_Diab[title].str.contains('No')) & clean_Diab.Diagnosis.str.contains('Negative')])
#     df = pd.DataFrame.from_dict(label_feature_relationship)
#     df = df.T
#     df.plot.bar(rot=0, title=title)
#     plt.ylabel('Counts')
#     # plt.show()
#
# label_feature_relationship = {}
# label_feature_relationship["Has Family History"] = {}
# label_feature_relationship["Has Family History"]['Positive'] = len(clean_Diab.loc[clean_Diab['Family History']==1 & clean_Diab.Diagnosis.str.contains('Positive')])
# label_feature_relationship["Has Family History"]['Negative'] = len(clean_Diab.loc[clean_Diab['Family History']==1 & clean_Diab.Diagnosis.str.contains('Negative')])
# label_feature_relationship["No Family History"] = {}
# label_feature_relationship["No Family History"]['Positive'] = len(clean_Diab.loc[clean_Diab['Family History']==0 & clean_Diab.Diagnosis.str.contains('Positive')])
# label_feature_relationship["No Family History"]['Negative'] = len(clean_Diab.loc[clean_Diab['Family History']==0 & clean_Diab.Diagnosis.str.contains('Negative')])
# df = pd.DataFrame.from_dict(label_feature_relationship)
# df = df.T
# df.plot.bar(rot=0, title='Family History')
# plt.ylabel('Counts')
# plt.show()
#
# #3c
# #1. Is there an age from which the chance of getting sick increases significantly?- A histogram is drawn showing a connection between age and a positive diagnosis:
# ax = clean_Diab.hist(column='Age', bins= 100, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
# ax = ax[0]
# for x in ax:
#     x.set_ylabel("Count", labelpad=20, weight='bold', size=12)
# plt.show()
#
# #2. Compare age histogram for positive diagnosis and age histogram for negative diagnosis:
# positive_age_series = clean_Diab[clean_Diab.Diagnosis.str.contains('Positive')]['Age']
# negative_age_series = clean_Diab[clean_Diab.Diagnosis.str.contains('Negative')]['Age']
# plt.hist(positive_age_series, bins=100, label='Positive Diagnosis')
# plt.hist(negative_age_series, bins=100, label='Negative Diagnosis')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.legend(loc='upper right')
# plt.show()
#
# #3. Pie chart of positive/negative diagnosis
# positive_count = 0
# negative_count = 0
# for idx, value in enumerate(clean_Diab['Diagnosis']):
#     if value =='Positive':
#         positive_count +=1
#     else:
#         negative_count +=1
# labels = ('Positive', 'Negative')
# sizes = [positive_count, negative_count]
# colors = ['lightcoral', 'yellowgreen']
# figureObject, axesObject = plt.subplots()
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.title("Positive and negative diagnosis percentages")
# plt.show()


#Question 4
#Encoding our data as one hot vectors
#not relevant and wrong!!
# X_binary = X.replace(['Yes','Female','Positive'],value = 1)
# X_binary = X_binary.replace(['No','Male','Negative'],value = 0)
# y_binary = y.replace(['Positive'],value = 1)
# X_binary = X_binary.drop('Age', axis=1)
# y_binary = y_binary.replace(['Negative'],value = 0)
#
# encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
# x_onehotvector = encoder.fit_transform(X_binary)
# LE = LabelEncoder()
# y_onehotvector = y_binary
# y_onehotvector = LE.fit_transform(np.ravel(y_onehotvector))

#relevant code- Sapir
#Encoding our data as one hot vectors
# X_binary = X.replace(['Yes','Female','Positive'],value = 1)
# X_binary = X_binary.replace(['No','Male','Negative'],value = 0)
# # X_binary = X_binary.drop('Age', axis=1)
# X_binary['Age'] = (X_binary['Age']-X_binary['Age'].mean())/X_binary['Age'].std()
# y_binary = y.replace(['Positive'],value = 1)
# y_binary = y_binary.replace(['Negative'],value = 0)
#
# x_onehotvector = X_binary
# y_onehotvector = y_binary

#Moran's suggestion
X_binary = 1* (X == 'Yes' & X == 'Female' & X =='Positive')
# #Question 5
# X_train, x_test, Y_train, y_test = train_test_split(x_onehotvector, y_onehotvector, test_size = 0.20, random_state = 0, stratify = y_onehotvector)
# # K cross fold+ SVM ( linear for 'svm_kernel':['linear'], non linear for 'svm_kernel':['rbf'])
# # Linear SVM model
# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
# svc = SVC(probability=True)
# C = np.array([0.001, 0.01, 1, 10, 100, 1000])
# pipe = Pipeline(steps=[('svm', svc)])
# svm_lin = GridSearchCV(estimator=pipe,
#              param_grid={'svm__kernel':['linear'], 'svm__C':C}, scoring=['roc_auc'],  cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
# svm_lin.fit(X_train, Y_train.ravel())
# best_svm_lin = svm_lin.best_estimator_
#
#
# calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
# calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
# calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
# calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]
#
# y_pred_test_lin = best_svm_lin.predict(x_test)
# y_pred_proba_test_lin = best_svm_lin.predict_proba(x_test)
# lin_loss = hinge_loss(y_test, y_pred_proba_test_lin[:,1])
# LSVM_score = roc_auc_score(y_test, y_pred_proba_test_lin[:,1])
#
# TN = calc_TN(y_test, y_pred_test_lin)
# FP = calc_FP(y_test, y_pred_test_lin)
# FN = calc_FN(y_test, y_pred_test_lin)
# TP = calc_TP(y_test, y_pred_test_lin)
# Se = TP/(TP+FN)
# Sp = TN/(TN+FP)
# PPV = TP/(TP+FP)
# NPV = TN/(TN+FN)
# Acc = (TP+TN)/(TP+TN+FP+FN)
# F1 = (2*Se*PPV)/(Se+PPV)
#
# #Non- linear SVM (RBF Kernel):
# pipe_rbf = Pipeline(steps=[('svm', svc)])
# svm_rbf = GridSearchCV(estimator=pipe,
#              param_grid={'svm__kernel':['rbf'], 'svm__C':C}, scoring=['roc_auc'],  cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
#
# svm_rbf.fit(X_train, Y_train.values.ravel())
# best_svm_rbf = svm_rbf.best_estimator_
#
# y_pred_test_rbf = best_svm_rbf.predict(x_test)
# y_pred_proba_test_rbf = best_svm_rbf.predict_proba(x_test)
# rbf_loss = hinge_loss(y_test, y_pred_proba_test_rbf[:,1])
# rbf_SVM_score = roc_auc_score(y_test, y_pred_proba_test_rbf[:,1])
#
# TN_nl = calc_TN(y_test, y_pred_test_rbf)
# FP_nl = calc_FP(y_test, y_pred_test_rbf)
# FN_nl = calc_FN(y_test, y_pred_test_rbf)
# TP_nl = calc_TP(y_test, y_pred_test_rbf)
# Se_nl = TP/(TP+FN)
# Sp_nl = TN/(TN+FP)
# PPV_nl = TP/(TP+FP)
# NPV_nl = TN/(TN+FN)
# Acc_nl = (TP+TN)/(TP+TN+FP+FN)
# F1_nl = (2*Se*PPV)/(Se+PPV)
#
# print("svm with  linear kernel:")
# print(f'Sensitivity is {Se:.2f}')
# print(f'Specificity is {Sp:.2f}')
# print(f'PPV is {PPV:.2f}')
# print(f'NPV is {NPV:.2f}')
# print(f'Accuracy is {Acc:.2f}')
# print(f'F1 is {F1:.2f}')
# print(f'The Linear Loss is {lin_loss:.2f}')
# print(f'AUC is {LSVM_score:.2f}')
# print("\n svm with rbf kernel:")
# print(f' Sensitivity is {Se_nl:.2f}')
# print(f' Specificity is {Sp_nl:.2f}')
# print(f' PPV is {PPV_nl:.2f}')
# print(f' NPV is {NPV_nl:.2f}')
# print(f' Accuracy is {Acc_nl:.2f}')
# print(f' F1 is {F1_nl:.2f}')
# print(f' Loss is {rbf_loss:.2f}')
# print(f' AUC is {rbf_SVM_score:.2f}')
#
# #Question 6- Feature Selection
# #Random Forest Network
# X_train, x_test, Y_train, y_test = train_test_split(x_onehotvector, y_onehotvector, test_size = 0.20, random_state = 0, stratify = y_onehotvector)
# clf = rfc(n_estimators=10)
# # scaler = StandardScaler()
# # X_train_scale = scaler.fit_transform(X_train)
# clf.fit(X_train, Y_train.values.ravel())
# w_ = clf.feature_importances_
# # w_positive = w_[::2]
#
# features=['Age','Gender','Increased Urination', 'Increased Thirst','Sudden Weight Loss','Weakness','Increased Hunger','Genital Thrush','Visual Blurring','Itching','Irritability','Delayed Healing','Partial Paresis','Muscle Stiffness','Hair Loss','Obesity','Family History']
# # features=['Male', 'Female', 'No Increased Urination', 'Increased Urination', 'No Increased Thirst', 'Increased Thirst', 'No Sudden Weight Loss', 'Sudden Weight Loss', 'No Weakness', 'Weakness', 'No Increased Hunger', 'Increased Hunger', 'No Genital Thrush', 'Genital Thrush', 'No Visual Blurring', 'Visual Blurring', 'No Itching', 'Itching', 'No Irritability', 'Irritability', 'No Delayed Healing', 'Delayed Healing', 'No Partial Paresis', 'Partial Paresis', 'No Muscle Stiffness', 'Muscle Stiffness', 'No Hair Loss', 'Hair Loss', 'No Obesity', 'Obesity', 'No Family History', 'Family History']
# # features=['Male', 'Female', 'No Increased Urination', 'Increased Urination', 'No Increased Thirst', 'Increased Thirst', 'No Sudden Weight Loss', 'Sudden Weight Loss', 'No Weakness', 'Weakness', 'No Increased Hunger', 'Increased Hunger', 'No Genital Thrush', 'Genital Thrush', 'No Visual Blurring', 'Visual Blurring', 'No Itching', 'Itching', 'No Irritability', 'Irritability', 'No Delayed Healing', 'Delayed Healing', 'No Partial Paresis', 'Partial Paresis', 'No Muscle Stiffness', 'Muscle Stiffness', 'No Hair Loss', 'Hair Loss', 'No Obesity', 'Obesity', 'No Family History', 'Family History']
# x = np.arange(len(features))
# x=np.ndarray.tolist(x)
# w_ = np.ndarray.tolist(w_)
# plt.bar(x, w_,0.5, color='c')
# plt.xticks(x,features,rotation=90);
# plt.ylabel("weights", fontsize=12)
# plt.title("Feature Weights- RFC")
# plt.show()
#
# # #Question 7
# #7a- PCA
# # scaler = StandardScaler()
# # X_train_scale = scaler.fit_transform(X_train)
# # x_test = scaler.transform(x_test)
# # x_test_binary = scaler.transform(x_test)
# n_components = 2
# pca = PCA(n_components = n_components, whiten= True)
# X_train_pca = pca.fit_transform(X_train)
# x_test_pca = pca.transform(x_test)
#
# def plt_2d_pca(X_pca,y):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, aspect='equal')
#     ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='b')
#     ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='r')
#     ax.legend(('Negative','Positive'))
#     ax.plot([0], [0], "ko")
#     ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
#     ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
#     ax.set_xlabel('$U_1$')
#     ax.set_ylabel('$U_2$')
#     ax.set_title('2D PCA')
#
# plt_2d_pca(x_test_pca,y_test)
# plt.show()
# #7c
# # Linear SVM model
# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
# svc = SVC(probability=True)
# C = np.array([0.001, 0.01, 1, 10, 100, 1000])
# pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
# svm_lin = GridSearchCV(estimator=pipe,
#              param_grid={'svm__kernel':['linear'], 'svm__C':C}, scoring=['roc_auc'],  cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
# svm_lin.fit(X_train_pca, Y_train)
# best_svm_lin = svm_lin.best_estimator_
#
# calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
# calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
# calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
# calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]
#
# y_pred_test_lin = best_svm_lin.predict(x_test_pca)
# y_pred_proba_test_lin = best_svm_lin.predict_proba(x_test_pca)
# lin_loss = hinge_loss(y_test.ravel(), y_pred_proba_test_lin[:,1])
# LSVM_score = roc_auc_score(y_test.ravel(), y_pred_proba_test_lin[:,1])
#
# TN = calc_TN(y_test, y_pred_test_lin)
# FP = calc_FP(y_test, y_pred_test_lin)
# FN = calc_FN(y_test, y_pred_test_lin)
# TP = calc_TP(y_test, y_pred_test_lin)
# Se = TP/(TP+FN)
# Sp = TN/(TN+FP)
# PPV = TP/(TP+FP)
# NPV = TN/(TN+FN)
# Acc = (TP+TN)/(TP+TN+FP+FN)
# F1 = (2*Se*PPV)/(Se+PPV)
#
# #Non- linear SVM (RBF Kernel):
# pipe_rbf = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
# svm_rbf = GridSearchCV(estimator=pipe,
#              param_grid={'svm__kernel':['rbf'], 'svm__C':C}, scoring=['roc_auc'],  cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
# svm_rbf.fit(X_train_pca, Y_train)
# best_svm_rbf = svm_rbf.best_estimator_
#
# y_pred_test_rbf = best_svm_rbf.predict(x_test_pca)
# y_pred_proba_test_rbf = best_svm_rbf.predict_proba(x_test_pca)
# rbf_loss = hinge_loss(y_test, y_pred_proba_test_rbf[:,1])
# rbf_SVM_score = roc_auc_score(y_test, y_pred_proba_test_rbf[:,1])
#
# TN_nl = calc_TN(y_test, y_pred_test_rbf)
# FP_nl = calc_FP(y_test, y_pred_test_rbf)
# FN_nl = calc_FN(y_test, y_pred_test_rbf)
# TP_nl = calc_TP(y_test, y_pred_test_rbf)
# Se_nl = TP/(TP+FN)
# Sp_nl = TN/(TN+FP)
# PPV_nl = TP/(TP+FP)
# NPV_nl = TN/(TN+FN)
# Acc_nl = (TP+TN)/(TP+TN+FP+FN)
# F1_nl = (2*Se*PPV)/(Se+PPV)
#
# print("svm with  linear kernel:")
# print(f'Sensitivity is {Se:.2f}')
# print(f'Specificity is {Sp:.2f}')
# print(f'PPV is {PPV:.2f}')
# print(f'NPV is {NPV:.2f}')
# print(f'Accuracy is {Acc:.2f}')
# print(f'F1 is {F1:.2f}')
# print(f'The Linear Loss is {lin_loss:.2f}')
# print(f'AUC is {LSVM_score:.2f}')
# print("\n svm with rbf kernel:")
# print(f' Sensitivity is {Se_nl:.2f}')
# print(f' Specificity is {Sp_nl:.2f}')
# print(f' PPV is {PPV_nl:.2f}')
# print(f' NPV is {NPV_nl:.2f}')
# print(f' Accuracy is {Acc_nl:.2f}')
# print(f' F1 is {F1_nl:.2f}')
# print(f' Loss is {rbf_loss:.2f}')
# print(f' AUC is {rbf_SVM_score:.2f}')
#
# #7d:Train the same models on the best two features from section 6: Increased Urination, Increased Thirst
# x_OHV_2feat = x_onehotvector[:,[3,5]]
# X_train, x_test, Y_train, y_test = train_test_split(x_OHV_2feat, y_onehotvector, test_size = 0.20, random_state = 0, stratify = y_onehotvector)
# # K cross fold+ SVM ( linear for 'svm_kernel':['linear'], non linear for 'svm_kernel':['rbf'])
# # Linear SVM model
# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
# svc = SVC(probability=True)
# C = np.array([0.001, 0.01, 1, 10, 100, 1000])
# pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
# svm_lin = GridSearchCV(estimator=pipe,
#              param_grid={'svm__kernel':['linear'], 'svm__C':C}, scoring=['roc_auc'],  cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
# svm_lin.fit(X_train, Y_train)
# best_svm_lin = svm_lin.best_estimator_
#
# calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
# calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
# calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
# calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]
#
# y_pred_test_lin = best_svm_lin.predict(x_test)
# y_pred_proba_test_lin = best_svm_lin.predict_proba(x_test)
# lin_loss = hinge_loss(y_test.ravel(), y_pred_proba_test_lin[:,1])
# LSVM_score = roc_auc_score(y_test.ravel(), y_pred_proba_test_lin[:,1])
#
# TN = calc_TN(y_test, y_pred_test_lin)
# FP = calc_FP(y_test, y_pred_test_lin)
# FN = calc_FN(y_test, y_pred_test_lin)
# TP = calc_TP(y_test, y_pred_test_lin)
# Se = TP/(TP+FN)
# Sp = TN/(TN+FP)
# PPV = TP/(TP+FP)
# NPV = TN/(TN+FN)
# Acc = (TP+TN)/(TP+TN+FP+FN)
# F1 = (2*Se*PPV)/(Se+PPV)
#
# #Non- linear SVM (RBF Kernel):
# pipe_rbf = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
# svm_rbf = GridSearchCV(estimator=pipe,
#              param_grid={'svm__kernel':['rbf'], 'svm__C':C}, scoring=['roc_auc'],  cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
#
# svm_rbf.fit(X_train, Y_train)
# best_svm_rbf = svm_rbf.best_estimator_
#
# y_pred_test_rbf = best_svm_rbf.predict(x_test)
# y_pred_proba_test_rbf = best_svm_rbf.predict_proba(x_test)
# rbf_loss = hinge_loss(y_test, y_pred_proba_test_rbf[:,1])
# rbf_SVM_score = roc_auc_score(y_test, y_pred_proba_test_rbf[:,1])
#
# TN_nl = calc_TN(y_test, y_pred_test_rbf)
# FP_nl = calc_FP(y_test, y_pred_test_rbf)
# FN_nl = calc_FN(y_test, y_pred_test_rbf)
# TP_nl = calc_TP(y_test, y_pred_test_rbf)
# Se_nl = TP/(TP+FN)
# Sp_nl = TN/(TN+FP)
# PPV_nl = TP/(TP+FP)
# NPV_nl = TN/(TN+FN)
# Acc_nl = (TP+TN)/(TP+TN+FP+FN)
# F1_nl = (2*Se*PPV)/(Se+PPV)
#
# print("svm with  linear kernel:")
# print(f'Sensitivity is {Se:.2f}')
# print(f'Specificity is {Sp:.2f}')
# print(f'PPV is {PPV:.2f}')
# print(f'NPV is {NPV:.2f}')
# print(f'Accuracy is {Acc:.2f}')
# print(f'F1 is {F1:.2f}')
# print(f'The Linear Loss is {lin_loss:.2f}')
# print(f'AUC is {LSVM_score:.2f}')
# print("\n svm with rbf kernel:")
# print(f' Sensitivity is {Se_nl:.2f}')
# print(f' Specificity is {Sp_nl:.2f}')
# print(f' PPV is {PPV_nl:.2f}')
# print(f' NPV is {NPV_nl:.2f}')
# print(f' Accuracy is {Acc_nl:.2f}')
# print(f' F1 is {F1_nl:.2f}')
# print(f' Loss is {rbf_loss:.2f}')
# print(f' AUC is {rbf_SVM_score:.2f}')







