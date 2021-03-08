import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

banknote = pd.read_csv("data/banknote.csv")
diabetes = pd.read_csv("data/diabetes.csv")
madelon = pd.read_csv("data/madelon.csv")
ozone = pd.read_csv("data/ozone.csv")
steel = pd.read_csv("data/steel.csv")


def drop_corr(X):
    df = pd.DataFrame(X)
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print(to_drop)
    df_drop = df.drop(to_drop, axis=1)
    return df_drop

    
banknote_drop = drop_corr(banknote)
banknote_drop.to_csv("data/banknote_prep.csv")

diabetes_drop = drop_corr(diabetes)
diabetes_drop.to_csv("data/banknote_prep.csv")

madelon_drop = drop_corr(madelon)
madelon_drop.to_csv("data/madelon_prep.csv")

ozone_drop = drop_corr(ozone)
ozone_drop.to_csv("data/ozone_prep.csv")

steel_drop = drop_corr(steel)
steel_drop.to_csv("data/steel_prep.csv")

