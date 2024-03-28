import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils as ut
# df=pd.read_csv('tp5_data/tp5_data1_train.txt',names=['x1','x2','y'])
# X_train=df[['x1','x2']]
# Y_train=np.array(df['y'])
# x1=np.ones((len(Y_train),1))
# X=np.array([x1,np.array(X_train['x1']).reshape(len(Y_train),1),np.array(X_train['x2']).reshape(len(Y_train),1)]).T
# X=X[0]
# tetamin=np.linalg.inv(X.T @ X) @ X.T@Y_train
# print(tetamin)
def prediction3(x):
    tetamin=[-0.39217817 ,0.03440471,-0.0277142 ]
    if tetamin[0]+tetamin[1]*x[0]+tetamin[2]*x[1]>0:
        return 1
    return 0
plt.figure(figsize=(12,8))
for label in np.unique(Y_train):
    plt.scatter(df[Y_train == label]['x1'], df[Y_train == label]['x2'], label=label, marker='+' if label == 0 else 'x')
plt.legend()
ut.plot_decision(X_train['x1'].min(),X_train['x1'].max(),X_train['x2'].min(),X_train['x2'].max(),prediction=prediction3)
plt.axis('equal')
plt.show()
