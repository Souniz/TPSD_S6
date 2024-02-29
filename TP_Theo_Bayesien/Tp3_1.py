import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
from tools import test
=======
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
>>>>>>> 0d99226 (Tp3 Theorie Bayesien)

df=pd.read_csv('tp1_data/tp1_data_train.txt')
df.columns=['x','y']
X_train=np.array(df['x'])
<<<<<<< HEAD
Y_train=np.array(df['y'])
=======
>>>>>>> 0d99226 (Tp3 Theorie Bayesien)
classe0=list(df[df['y']==0]['x'])
classe1=list(df[df['y']==1]['x'])
Dmin=min([i for i in classe0 if i>= min(classe1)])
Dmax=max([i for i in classe0 if i>= min(classe1)])
<<<<<<< HEAD
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
=======
subdivision=np.arange(Dmin,Dmax,0.1)
seuil=min([test2('tp1_data/tp1_data_valid.txt', frontiere) for frontiere in subdivision])
frontiere=[i for i in subdivision if test2('tp1_data/tp1_data_valid.txt', i)==seuil][0]
frontier=np.around(frontiere,4)
print(frontiere)
>>>>>>> 0d99226 (Tp3 Theorie Bayesien)

