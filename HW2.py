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

        a = pd.to_numeric(CTG_features[col], errors='coerce')
        bank_for_col = a.dropna()
        bank_for_col = np.random.choice(bank_for_col, size=len(a))
        c_cdf[col] = a.fillna(pd.Series(bank_for_col))

    return df

clean_Diab = nan2value_random(Diab)