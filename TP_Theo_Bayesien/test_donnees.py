from tools import show,calcul_frontiere,test
from pathlib import Path
dossier=r'C:\Users\Souniz\Desktop\L3_SD\TP_S6\TP_Theo_Bayesien\tp3_data'
p=Path(dossier)
data_train=[]
data_valid=[]
fichier=list(p.glob('*.txt'))
for i in range(0,len(fichier)-1,2):
    data_train.append(fichier[i].absolute())
    data_valid.append(fichier[i+1].absolute())

for i in range(len(data_train)):
    print(f'------- data numero {i+1} -----')
    frontiere=calcul_frontiere(str(data_train[i]),str(data_valid[i]))
    taux,matrice=test(str(data_valid[i]),frontiere)
    print(f"Le taux d'errer est {taux}")
    print(f"La matrice de confusion est  {matrice}")