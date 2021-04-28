from dev.model_selection import TransformerFromHyperP
import pandas as pd
import numpy as np
from joblib import dump, load

from sklearn.base import BaseEstimator, TransformerMixin

clf = load("model/logistic_fyp.joblib")
df = pd.read_csv("../datasets/prem_final_datasets/prem_test_set.csv")


lvp_nw = df.loc[[0]]
mutd_crys = df.loc[[23]]

df = df.drop(["Date", "Home", "Score", "Away", "Venue", "MP_Home", "MP_Away"], axis=1)

per_90_features = ["GlsPer90_Home", "AstPer90_Home", "Sh/90_Home", "SoT/90_Home", "GA90_Home",
                   "GlsPer90_Away", "AstPer90_Away", "Sh/90_Away", "SoT/90_Away", "GA90_Away",
                   "GlsPer90_2_Home", "AstPer90_2_Home", "GlsPer90_3_Home", "AstPer90_3_Home", "GlsPer90_4_Home",
                   "AstPer90_4_Home", "GlsPer90_5_Home", "AstPer90_5_Home", "GlsPer90_6_Home", "AstPer90_6_Home",
                   "GlsPer90_7_Home", "AstPer90_7_Home", "GlsPer90_8_Home", "AstPer90_8_Home",
                   "GlsPer90_9_Home", "AstPer90_9_Home", "GlsPer90_10_Home", "AstPer90_10_Home",
                   "GlsPer90_11_Home", "AstPer90_11_Home", "GlsPer90_2_Away", "AstPer90_2_Away",
                   "GlsPer90_3_Away", "AstPer90_3_Away", "GlsPer90_4_Away", "AstPer90_4_Away",
                   "GlsPer90_5_Away", "AstPer90_5_Away", "GlsPer90_6_Away", "AstPer90_6_Away",
                   "GlsPer90_7_Away", "AstPer90_7_Away", "GlsPer90_8_Away", "AstPer90_8_Away",
                   "GlsPer90_9_Away", "AstPer90_9_Away", "GlsPer90_10_Away", "AstPer90_10_Away",
                   "GlsPer90_11_Away", "AstPer90_11_Away"]

df = df.drop(per_90_features, axis=1)
redundant_features = ["CrdR_Home", "D_Away", "CrdR_Away", "Sh_Away", "MP_2_Home", "Min_2_Home", "MP_4_Home",
                      "Min_4_Home", "MP_5_Home", "Min_5_Home", "MP_6_Home", "Min_6_Home", "MP_8_Home", "Min_8_Home",
                      "MP_11_Home", "Min_11_Home", "MP_2_Away", "Min_2_Away"]

df = df.drop(redundant_features, axis=1)

lvp_nw_pred = clf.predict(lvp_nw)
mutd_crys_pred = clf.predict(mutd_crys)

print('''
%s\t%s\t%s\t%s\t%s
%s\t%s\t%s\t%s\t%s

''' % (lvp_nw["Home"].item(), lvp_nw["Away"].item(), lvp_nw["Score"].item(), lvp_nw["Result"].item(), lvp_nw_pred,
       mutd_crys["Home"].item(), mutd_crys["Away"].item(), mutd_crys["Score"].item(), mutd_crys["Result"].item(), mutd_crys_pred))
