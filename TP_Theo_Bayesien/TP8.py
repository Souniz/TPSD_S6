import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils as ut

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils as ut

df=pd.read_csv('tp8_data/tp8_data1_train.txt',names=['x1','x2','y'])
X_train=df[['x1','x2']]
Y_train=np.array(df['y'])
X_classe0=df[df['y']==0]
X_classe0=X_classe0[['x1','x2']]
X_classe1=df[df['y']==1]
X_classe1=X_classe1[['x1','x2']]
couleur={1:'blue',0:'orange'}



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
invCov0=np.linalg.inv(cov0)
detCov0=np.linalg.det(cov0)
detCov1=np.linalg.det(cov1)
invCov1=np.linalg.inv(cov1)
p0=len(X_classe0)/len(df)
p1=len(X_classe1)/len(df)
def predictionMahanlobi(x):
    if (x-u0).T @ invCov0  @ (x-u0)+np.log(detCov0)-2*np.log(p0)<(x-u1).T @ invCov1  @ (x-u1) +np.log(detCov1)-2*np.log(p1):
            return 0
    return 1


def prediction(x):
    if(x-u0)@(x-u0).T<(x-u1)@(x-u1).T:
            return 0
    return 1
ddf=pd.read_csv('tp8_data/tp8_data1_valid.txt',names=['x1','x2','y'])
X_valid=df[['x1','x2']]
Y_valid=np.array(df['y'])
couleur={1:'blue',0:'orange'}
plt.figure(figsize=(12,8))
for label in np.unique(Y_valid):
    plt.scatter(df[Y_valid == label]['x1'], df[Y_valid == label]['x2'], label=label, marker='+' if label == 0 else 'x')
plt.legend()
ut.plot_decision(X_valid['x1'].min(),X_valid['x1'].max(),X_train['x2'].min(),X_valid['x2'].max(),prediction=predictionMahanlobi)
plt.axis('equal')
plt.show()
y_pred=[prediction(i) for i in np.array(X_valid)]
mat_confusion2,Taux2=ut.create_mat(y_pred,Y_valid)
print(f"{Taux2} %")
print(mat_confusion2)




















