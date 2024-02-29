import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def show(fichier):
    df=pd.read_csv(fichier)
    df.columns=['x','y']
    print(f"L'effectifs de la classe 0 est {len(df[df['y']==0])}")
    print(f"L'effectifs de la classe 1 est {len(df[df['y']==1])}")
    plt.hist(df[df['y']==0]['x'],color='blue',label='Classe 0',alpha=0.7)
    plt.hist(df[df['y']==1]['x'],color='red',label='Classe 1',alpha=0.7)
    plt.legend()
    plt.show()

def test(validation, frontiere):
    df_valid=pd.read_csv(validation)
    df_valid.columns=['x','y']
    X_valid=np.array(df_valid['x'])
    y_pred=[0 if i<frontiere else 1 for i in X_valid]
    y_valid=np.array(df_valid['y'])
    errr=[1 for i in range(0,len(y_valid)) if y_valid[i]!=y_pred[i]]
    nb_errer=len(errr)
    Taux=nb_errer/len(y_pred)*100
    print(f"Le taux d'errer est {Taux}")
    postif_positif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==0 and y_valid[i]==0])
    postif_negatif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==0 and y_valid[i]==1])
    negatif_negatif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==1 and y_valid[i]==0])
    negatif_positif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==1 and y_valid[i]==1])
    mat_conf=np.array([[postif_positif,postif_negatif],[negatif_negatif,negatif_positif]])
    print(f"La matrice de confusion est {mat_conf}")

