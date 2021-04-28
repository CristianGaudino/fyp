import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


path_15 = "../datasets/odds/bookie_odds_15.csv"
path_16 = "../datasets/odds/bookie_odds_16.csv"
path_17 = "../datasets/odds/bookie_odds_17.csv"
path_18 = "../datasets/odds/bookie_odds_18.csv"
# path_19 = "../datasets/odds/bookie_odds_19.csv"

file_15 = open(path_15, "r", encoding="utf-8")
file_16 = open(path_16, "r", encoding="utf-8")
file_17 = open(path_17, "r", encoding="utf-8")
file_18 = open(path_18, "r", encoding="utf-8")
# file_19 = open(path_19, "r", encoding="utf-8")

files = [file_15, file_16, file_17, file_18]

ftr = []
ftr_pred = []
b365_indices = [23, 24, 25]

for file in files:
    for line in file.readlines()[1:]:   # skip the first line
        line = line.split(",")
        ftr.append(line[6])

        b365 = [line[i] for i in b365_indices]

        if b365[0] == min(b365):
            ftr_pred.append("H")
        elif b365[1] == min(b365):
            ftr_pred.append("D")
        else:
            ftr_pred.append("A")


''' ---- Accuracy ---- '''
count = 0
correct = 0
for result, result_pred in zip(ftr, ftr_pred):
    count += 1
    if result == result_pred:
        correct += 1

accuracy = (correct / count) * 100
print("Total: %d \nCorrect: %d \nAccuracy: %.3f" % (count, correct, accuracy))


''' ---- Confusion Matrix ---- '''
y_actu = pd.Series(ftr, name='Actual')
y_pred = pd.Series(ftr_pred, name='Predicted')

print(pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))
'''
Total: 1520 
Correct: 845 
Accuracy: 55.592
Predicted    A     H   All
Actual                    
A          265   196   461
D           99   262   361
H          118   580   698
All        482  1038  1520


Total: 380 
Correct: 223 
Accuracy: 58.684
Predicted    A    H  All
Actual                  
A           78   50  128
D           18   53   71
H           36  145  181
All        132  248  380
'''