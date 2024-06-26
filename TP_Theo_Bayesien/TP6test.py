import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils as ut

df=pd.read_csv('tp5_data/tp5_data2_train.txt',names=['x1','x2','y'])
X_train=np.array(df[['x1','x2']])
Y_train=np.array(df['y'])

def calcul_parametre(df):
    X_classe0=df[df['y']==0]
    X_classe0=X_classe0[['x1','x2']]
    X_classe1=df[df['y']==1]
    X_classe1=X_classe1[['x1','x2']]
    moyenne0=[X_classe0['x1'].mean(),X_classe0['x2'].mean()]
    cov0=(X_classe0-moyenne0).T@(X_classe0-moyenne0)
    cov0=cov0/len(X_classe0['x1'])
    moyenne1=[X_classe1['x1'].mean(),X_classe1['x2'].mean()]
    cov1=(X_classe1-moyenne1).T@(X_classe1-moyenne1)
    cov1=cov1/len(X_classe1['x1'])
    return moyenne0,cov0,moyenne1,cov1


u0,cov0,u1,cov1=calcul_parametre(df)
def prediction(x):
    """
      x:represente la donnees a predire
      u0: represente la moyenne de la classe 0
      u1:represente la moyenne de la classe 0
      La fonction retourne 0 si la distance euclidienne entre x et la moyenne de la 
      classe 0 est inferieur a celle de la classe 1 et retourne 1 siono
    """
    if(x-u0)@(x-u0).T<(x-u1)@(x-u1).T:
            return 0
    return 1



df=pd.read_csv('tp5_data/tp5_data2_valid.txt',names=['x1','x2','y'])
X_train=df[['x1','x2']]
Y_train=df['y']
y_pred=[prediction(i) for i in np.array(X_train)]
mat_confusion1,Taux1=ut.create_mat(y_pred,Y_train)
print(Taux1)
print(mat_confusion1)
for label in np.unique(Y_train):
    plt.scatter(df[Y_train == label]['x1'], df[Y_train == label]['x2'], label=label, marker='+' if label == 0 else 'x')
plt.legend()
ut.plot_decision(X_train['x1'].min(),X_train['x1'].max(),X_train['x2'].min(),X_train['x2'].max(),prediction=prediction)
plt.axis('equal')
plt.show()