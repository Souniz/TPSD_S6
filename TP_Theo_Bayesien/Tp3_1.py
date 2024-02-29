import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import  show
def test2(validation, frontiere):
    df_valid=pd.read_csv(validation)
    df_valid.columns=['x','y']
    X_valid=np.array(df_valid['x'])
    y_pred=[0 if i<frontiere else 1 for i in X_valid]
    y_valid=np.array(df_valid['y'])
    errr=[1 for i in range(0,len(y_valid)) if y_valid[i]!=y_pred[i]]
    nb_errer=len(errr)
    Taux=nb_errer/len(y_pred)*100
    return np.around(Taux,4)
#show('tp1_data/tp1_data_train.txt')

df=pd.read_csv('tp1_data/tp1_data_train.txt')
df.columns=['x','y']
X_train=np.array(df['x'])
classe0=list(df[df['y']==0]['x'])
classe1=list(df[df['y']==1]['x'])
Dmin=min([i for i in classe0 if i>= min(classe1)])
Dmax=max([i for i in classe0 if i>= min(classe1)])
subdivision=np.arange(Dmin,Dmax,0.1)
seuil=min([test2('tp1_data/tp1_data_valid.txt', frontiere) for frontiere in subdivision])
frontiere=[i for i in subdivision if test2('tp1_data/tp1_data_valid.txt', i)==seuil][0]
frontier=np.around(frontiere,4)
print(frontiere)

