import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('tp1_data/tp1_data_train.txt')

df.columns=['x','y']
X_train=np.array(df['x'])
Y_train=np.array(df['y'])
#3) Combien de donn´ees sont dans la classe 0 ? dans la classe 1 ?
nb_class0=len(df[df['y']==0])
nb_class1=len(df[df['y']==1])
# plt.hist(df[df['y']==0]['x'],color='blue',label='Classe 0')
# plt.hist(df[df['y']==1]['x'],color='red',label='Classe 1')
# plt.legend()
Delta=145 #seuil
def prediction(x):
    """_summary_

    Args:
        x (float): une entree

    Returns:
        binaire: la classe correspondant
    """
    return 0 if x<Delta else 1


#3 Phase de validation
df_valid=pd.read_csv('tp1_data/tp1_data_valid.txt')
df_valid.columns=['x','y']
X_valid=np.array(df_valid['x'])
y_pred=[prediction(i) for i in X_valid]
y_valid=np.array(df_valid['y'])
##nombre d'errer
errr=[1 for i in range(0,len(y_valid)) if y_valid[i]!=y_pred[i]]
nb_errer=len(errr)
Taux=nb_errer/len(y_pred)*100
postif_positif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==0 and y_valid[i]==0])
postif_negatif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==0 and y_valid[i]==1])
negatif_negatif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==1 and y_valid[i]==1])
negatif_positif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==1 and y_valid[i]==0])
mat_conf=np.array([[postif_positif,postif_negatif],[negatif_positif,negatif_negatif]])
print(mat_conf)



