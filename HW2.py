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
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier as rfc
import random
from pathlib import Path
import sys
import matplotlib as mpl
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
#from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline




#Question 1

file = Path.cwd().joinpath('HW2_data.csv')
Diab = pd.read_csv(file)

def nan2value_random(df):

    df = df.dropna(axis=0, thresh=15) #Of the 565 patients, we want to remove patients with at least 3 NaN.
    # Patients with 2 nan values or less: we will have the NaNs replaced with random values from a feature values (column).
    # Since this is a small number of patients from the group examined, if a bias is created it is very small, and yet the given data is large enough.
    for col in df:

        bank_for_col = df[col].dropna()
        bank_for_col = np.random.choice(bank_for_col, size=len(df[col]))
        df[col] = df[col].fillna(pd.Series(bank_for_col))


    return df
clean_Diab = nan2value_random(Diab)

#Question 2

X = clean_Diab
y= X[['Diagnosis']]
# X = X.to_numpy() # can also be X.values
# y = y.to_numpy() # can also be y.values
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10, stratify=y)


#Question 3

# 3a
#create binary train, test DataFrames (except for age feature)
# X_train = pd.DataFrame(X_train, columns=['Gender', 'Increased Urination', 'Increased Thirst', 'Sudden Weight Loss', 'Weakness', 'Increased Hunger', 'Genital Thrush', 'Visual Blurring', 'Itching', 'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Hair Loss', 'Obesity', 'Diagnosis', 'Family History'])
X_train_binary = X_train.replace(['Yes','Female','Positive'],value = 1)
X_train_binary = X_train_binary.replace(['No','Male','Negative'],value = 0)
x_test_binary = x_test.replace(['Yes','Female','Positive'],value = 1)
x_test_binary = x_test_binary.replace(['No','Male','Negative'],value = 0)
#create a dictionary with features and values as %train, %test, %delta
list_train = [None]*17
list_test = [None]*17
delta = [None]*17
features_dictionary={}
features_dictionary['Positive Feature']=['%Train', '%Test', '%Delta']
for i in range(0,17):
    list_train[i] = X_train_binary.iloc[:,i+1].sum()*(100/len(X_train_binary))
    list_test[i] = x_test_binary.iloc[:,i+1].sum()*(100/len(x_test_binary))
    delta[i] = abs(list_train[i]-list_test[i])
    features_dictionary[clean_Diab.columns[i+1]]= [list_train[i], list_test[i], delta[i]]

df_features_dictionary = pd.DataFrame(features_dictionary).transpose()
print(df_features_dictionary)

#3b
#create a dictionary that shows the relationship between feature and label.
#keys: 1.positive feature 2.negative feature
#values for each key: 1. number of positive diagnosis 2.number of negative diagnosis
label_feature_relationship={}
label_feature_relationship['Female'] = {}
label_feature_relationship['Female']['Positive'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Female')) & (clean_Diab.Diagnosis.str.contains('Positive'))])
label_feature_relationship['Female']['Negative'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Female')) & (clean_Diab.Diagnosis.str.contains('Negative'))])

label_feature_relationship['Male'] = {}
label_feature_relationship['Male']['Positive'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Male')) & (clean_Diab.Diagnosis.str.contains('Positive'))])
label_feature_relationship['Male']['Negative'] = len(clean_Diab[(clean_Diab.Gender.str.contains('Male')) & (clean_Diab.Diagnosis.str.contains('Negative'))])

for i in range(2,16):
    title = clean_Diab.columns[i]
    label_feature_relationship["Has %s" %title] = {}
    label_feature_relationship["Has %s" %title]['Positive'] = len(clean_Diab[(clean_Diab[title].str.contains('Yes')) & clean_Diab.Diagnosis.str.contains('Positive')])
    label_feature_relationship["Has %s" %title]['Negative'] = len(clean_Diab[(clean_Diab[title].str.contains('Yes')) & clean_Diab.Diagnosis.str.contains('Negative')])
    label_feature_relationship["No %s" % title] = {}
    label_feature_relationship["No %s" % title]['Positive'] = len(clean_Diab[(clean_Diab[title].str.contains('No')) & clean_Diab.Diagnosis.str.contains('Positive')])
    label_feature_relationship["No %s" % title]['Negative'] = len(clean_Diab[(clean_Diab[title].str.contains('No')) & clean_Diab.Diagnosis.str.contains('Negative')])

label_feature_relationship["Has Family History"] = {}
label_feature_relationship["Has Family History"]['Positive'] = len(clean_Diab.loc[clean_Diab['Family History']==1 & clean_Diab.Diagnosis.str.contains('Positive')])
label_feature_relationship["Has Family History"]['Negative'] = len(clean_Diab.loc[clean_Diab['Family History']==1 & clean_Diab.Diagnosis.str.contains('Negative')])
label_feature_relationship["No Family History"] = {}
label_feature_relationship["No Family History"]['Positive'] = len(clean_Diab.loc[clean_Diab['Family History']==0 & clean_Diab.Diagnosis.str.contains('Positive')])
label_feature_relationship["No Family History"]['Negative'] = len(clean_Diab.loc[clean_Diab['Family History']==0 & clean_Diab.Diagnosis.str.contains('Negative')])

#plotting:
# pd.DataFrame(label_feature_relationship).T.plot(kind='bar')
for dict, value_dict in label_feature_relationship.items():
        df = pd.DataFrame.from_dict(value_dict)
        df_trans = df.T
        df_trans.plot.bar(rot=0, title=feature)
        plt.ylabel('Counts')

#3c
#pd.plotting.scatter_matrix(clean_Diab[['Gender','Increased Urination','Increased Thirst','Sudden Weight Loss']])
#plt.show()


#Question 4

Y_train = 1 * (Y_train=='M')
y_test = 1 * (y_test=='M')

#Question 5

#WHERE IS THE FOLD?

#linear model
#logistic regrssion
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


logreg = LogisticRegression(solver='saga', penalty='none', max_iter=10000, multi_class='multinomial')
logreg = logreg.fit(X_train, Y_train)
y_pred_log = logreg.predict(x_test)
y_pred_log_train = logreg.predict(X_train)
w_log = logreg.coef_
y_pred_log = logreg.predict_proba(X_test)

print("AUC Test is : " + str("{0:.2f}".format(100 * metrics.roc_auc_score(y_test, y_pred_log))) + "%")
print("AUC Train is : " + str("{0:.2f}".format(100 * metrics.roc_auc_score(Y_train, y_pred_log_train))) + "%")
print("F1 Test score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred_log, average='macro'))) + "%")
print("F1 Train score is: " + str("{0:.2f}".format(100 * metrics.f1_score(Y_train, y_pred_log_train, average='macro'))) + "%")
print("LOSS Test score is: " + str("{0:.2f}".format(100 * metrics.log_loss(y_test, y_pred_log, average='macro'))) + "%")
print("LOSS Train score is: " + str("{0:.2f}".format(100 * metrics.log_loss(Y_train, y_pred_log_train, average='macro'))) + "%")
print(" Test Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred_log))) + "%")
print(" Train Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(Y_train_test, y_pred_log_train))) + "%")



#nonlinear
from sklearn import svm
from sklearn.model_selection import GridSearchCV

svm_nonlinear = svm.SVC(kernel='rbf')
svm_nonlinear.fit(X_train, X_train)
y_test_nonlinear = svm_nonlinear.predict(x_test)
y_pred_nonlinear_train = svm_nonlinear.predict_proba(X_train)

print("AUC(NL) Test is : " + str("{0:.2f}".format(100 * metrics.roc_auc_score(y_test, y_test_nonlinear))) + "%")
print("AUC(NL) Train is : " + str("{0:.2f}".format(100 * metrics.roc_auc_score(Y_train, y_pred_nonlinear_train))) + "%")
print("F1(NL) Test score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_test_nonlinear, average='macro'))) + "%")
print("F1(NL) Train score is: " + str("{0:.2f}".format(100 * metrics.f1_score(Y_train, y_pred_nonlinear_train, average='macro'))) + "%")
print("LOSS Test(NL) score is: " + str("{0:.2f}".format(100 * metrics.log_loss(y_test, y_test_nonlinear, average='macro'))) + "%")
print("LOSS Train(NL) score is: " + str("{0:.2f}".format(100 * metrics.log_loss(Y_train,y_pred_nonlinear_train, average='macro'))) + "%")
print(" Test Accuracy(NL) is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test,y_test_nonlinear))) + "%")
print(" Train Accuracy(NL) is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(Y_train, y_pred_nonlinear_train))) + "%")
