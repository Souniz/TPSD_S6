import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils as ut
from TP5 import prediction

#4 Test de la fonction de prediction
df=pd.read_csv('tp5_data/tp5_data1_valid.txt',names=['x1','x2','y'])
X_train=np.array(df[['x1','x2']])
Y_train=np.array(df['y'])
y_pred=[prediction(i) for i in X_train]
mat_confusion1,Taux1=ut.create_mat(y_pred,Y_train)
print(Taux1)
print(mat_confusion1)