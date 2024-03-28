import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils_TP9 as ut


df=pd.read_csv('tp9_data/tp9_data_train.txt',names=['x1','x2','y'])
X_train=df[['x1','x2']]
Y_train=np.array(df['y'])
# X_classe0=df[df['y']==0]
# X_classe0=X_classe0[['x1','x2']]
# X_classe1=df[df['y']==1]
# X_classe1=X_classe1[['x1','x2']]
couleur={1:'blue',0:'orange',2:'red',3:'cyan',4:'marron'}
def calcul_parametre(df,classe):
    X_classe=df[df['y']==classe]
    X_classe=X_classe[['x1','x2']]
    moyenne=[X_classe['x1'].mean(),X_classe['x2'].mean()]
    cov=(X_classe-moyenne).T@(X_classe-moyenne)
    cov=cov/len(X_classe['x1'])
    return moyenne,cov

def predictionMahanlobi(x):
    dist=[]
    for i in range(5):
        u,cov=calcul_parametre(df,i)
        detCov=np.linalg.det(cov)
        invCov=np.linalg.inv(cov)
        p=len(df[df['y']==i])/len(df)
        invCov=np.linalg.inv(cov)
        dist.append((x-u).T @ invCov @ (x-u)+np.log(detCov)-2*np.log(p))
    dist=np.array(dist)
    return np.argmin(dist)


plt.figure(figsize=(12,8))
for label in np.unique(Y_train):
    plt.scatter(df[Y_train == label]['x1'], df[Y_train == label]['x2'], label=label, marker='+' if label == 0 else 'x')
plt.legend()
ut.plot_decision_multi(X_train['x1'].min(),X_train['x1'].max(),X_train['x2'].min(),X_train['x2'].max(),prediction=predictionMahanlobi)
plt.axis('equal')
plt.show()






















