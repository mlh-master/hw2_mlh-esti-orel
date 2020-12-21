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
#3c
pd.plotting.scatter_matrix(clean_Diab_int[['Age','Gender','Increased Urination','Increased Thirst','Sudden Weight Loss']])
plt.show()
#Question 4

Y_train = 1 * (Y_train=='M')
y_test = 1 * (y_test=='M')

#Question 5

