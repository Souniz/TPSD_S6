import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def show(fichier):
    df=pd.read_csv(fichier)
    df.columns=['x','y']
    plt.hist(df[df['y']==0]['x'],color='blue',label='Classe 0',alpha=0.7)
    plt.hist(df[df['y']==1]['x'],color='red',label='Classe 1',alpha=0.7)
    plt.legend()
    plt.show()
def calcul_frontiere(fichier_train,fichier_valid):
    """Calcul la frontiere qui donne le plus petit taux d'errer

    Args:
        fichier_train (str): le fichier des donnees d'entrainement
        fichier_valid (str): le fichier de validation
    """
    df=pd.read_csv(fichier_train)
    df.columns=['x','y']
    classe0=list(df[df['y']==0]['x'])
    classe1=list(df[df['y']==1]['x'])
    Dmin=min([i for i in classe0 if i>= min(classe1)])
    Dmax=max([i for i in classe0 if i>= min(classe1)])
    subdivision=np.arange(Dmin,Dmax,0.1)
    seuil=np.array([test(fichier_valid, frontiere)[0] for frontiere in subdivision])
    frontiere=subdivision[np.argmin(seuil)]
    frontiere=np.around(frontiere,4)
    return frontiere


def test(validation, frontiere):
    """Prend un fichier de validation et une frontiere 
       et calcule puis retourne le taux d'errer et la matrice de confusion
    Args:
        validation (str): le ficher de validation
        frontiere (float): le frontiere

    Returns:
        tuple: le taux d'errer et la matrice de confusion
    """
    df_valid=pd.read_csv(validation)
    df_valid.columns=['x','y']
    X_valid=np.array(df_valid['x'])
    y_pred=[0 if i<frontiere else 1 for i in X_valid]
    y_valid=np.array(df_valid['y'])
    errr=[1 for i in range(0,len(y_valid)) if y_valid[i]!=y_pred[i]]
    nb_errer=len(errr)
    Taux=nb_errer/len(y_pred)*100
    postif_positif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==0 and y_valid[i]==0])
    postif_negatif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==0 and y_valid[i]==1])
    negatif_negatif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==1 and y_valid[i]==1])
    negatif_positif=len([1 for i in range(0,len(y_valid)) if y_pred[i]==1 and y_valid[i]==0])
    mat_conf=np.array([[postif_positif,postif_negatif],[negatif_positif,negatif_negatif]])
    return Taux,mat_conf

