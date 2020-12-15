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

#Question 2
clean_Diab = nan2value_random(Diab)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(clean_Diab, np.ravel(clean_Diab), test_size=0.2, random_state=0, stratify=np.ravel(clean_Diab))

X = clean_Diab
y= X[['Diagnosis']]
X = X.to_numpy() # can also be X.values
y = y.to_numpy() # can also be y.values
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10, stratify=y)
