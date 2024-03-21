import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import utils as ut
df=pd.read_csv('tp5_data/tp5_data1_valid.txt',names=['x1','x2','y'])
X_train=df[['x1','x2']]
Y_train=np.array(df['y'])
couleur={1:'blue',0:'orange'}
def prediction(x):
    #f(x)=x-20
    if x[1]>x[0]-20:
        return 0
    else:
        return 1
plt.figure(figsize=(16,8))
for label in np.unique(Y_train):
    plt.scatter(df[Y_train == label]['x1'], df[Y_train == label]['x2'], label=label, marker='+' if label == 0 else 'x')
plt.legend()
ut.plot_decision(X_train['x1'].min(),X_train['x1'].max(),X_train['x2'].min(),X_train['x2'].max(),prediction=prediction)
plt.show()

