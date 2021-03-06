import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

banknote = pd.read_csv("data/banknote.csv")
diabetes = pd.read_csv("data/diabetes.csv")
madelon = pd.read_csv("data/madelon.csv")
ozone = pd.read_csv("data/ozone.csv")
steel = pd.read_csv("data/steel.csv")


def reg(X):
    mms = MinMaxScaler()
    X.iloc[:, :-1] = mms.fit_transform(X.iloc[:, :-1])
    le = LabelEncoder()
    X.iloc[:, -1] = le.fit_transform(X.iloc[:, -1])
    return X


def drop_corr(df):
    df = pd.DataFrame(reg(df))
    # cor_matrix = df.corr().abs()
    # upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    # to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    # print(to_drop)
    # df_drop = df.drop(to_drop, axis=1)
    # return df_drop
    X = df[list(set(df.columns) - {df.columns[-1]})]

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]
    to_drop = vif_data[vif_data["VIF"] > 13].feature.values
    print(sorted(to_drop))
    print(vif_data)
    df_drop = df.drop(to_drop, axis=1)
    return df_drop

banknote_drop = drop_corr(banknote)
banknote_drop_train, banknote_drop_test = train_test_split(banknote_drop, test_size=0.33, random_state=123)
banknote_drop_train.to_csv("data/banknote_train.csv")
banknote_drop_test.to_csv("data/banknote_test.csv")

diabetes_drop = drop_corr(diabetes)
diabetes_drop_train, diabetes_drop_test = train_test_split(diabetes_drop, test_size=0.33, random_state=123)
diabetes_drop_train.to_csv("data/diabetes_train.csv")
diabetes_drop_test.to_csv("data/diabetes_test.csv")

madelon_drop = drop_corr(madelon)
madelon_drop_train, madelon_drop_test = train_test_split(madelon_drop, test_size=0.33, random_state=123)
madelon_drop_train.to_csv("data/madelon_train.csv")
madelon_drop_test.to_csv("data/madelon_test.csv")

ozone_drop = drop_corr(ozone)
ozone_drop_train, ozone_drop_test = train_test_split(ozone_drop, test_size=0.33, random_state=123)
ozone_drop_train.to_csv("data/ozone_train.csv")
ozone_drop_test.to_csv("data/ozone_test.csv")

steel_drop = drop_corr(steel)
steel_drop_train, steel_drop_test = train_test_split(steel_drop, test_size=0.33, random_state=123)
steel_drop_train.to_csv("data/steel_train.csv")
steel_drop_test.to_csv("data/steel_test.csv")

