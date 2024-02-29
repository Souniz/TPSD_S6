import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import test

df=pd.read_csv('tp1_data/tp1_data_train.txt')
df.columns=['x','y']
X_train=np.array(df['x'])
Y_train=np.array(df['y'])
classe0=list(df[df['y']==0]['x'])
classe1=list(df[df['y']==1]['x'])
Dmin=min([i for i in classe0 if i>= min(classe1)])
Dmax=max([i for i in classe0 if i>= min(classe1)])
subdivision=np.arange(Dmin,Dmax,0.01)
seuil=np.array([test('tp1_data/tp1_data_valid.txt', frontiere)[0] for frontiere in subdivision])
frontiere=subdivision[np.argmin(seuil)]
frontiere=np.around(frontiere,4)
df_valid=pd.read_csv('tp1_data/tp1_data_valid.txt')
df_valid.columns=['x','y']
y_predi=[0 if i<frontiere else 1 for i in list(df_valid['x'])]
y_valid=df_valid['y'].values
positif_positif=len([0 for i in range(len(y_predi)) if y_predi[i]==0 and y_valid[i]==0])
negatif_negatif=len([0 for i in range(len(y_predi)) if y_predi[i]==1 and y_valid[i]==1])
positif_negatif=len([0 for i in range(len(y_predi)) if y_predi[i]==0 and y_valid[i]==1])
negatif_positif=len([0 for i in range(len(y_predi)) if y_predi[i]==1 and y_valid[i]==0])
mat_conf=np.array([[positif_positif,positif_negatif],[negatif_positif,negatif_negatif]])
print(mat_conf)

