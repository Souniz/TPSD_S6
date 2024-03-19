from tools import show2,calcul_frontiere,test
from pathlib import Path
import matplotlib.pyplot as plt
dossier=r'C:\Users\Souniz\Desktop\L3_SD\TP_S6\TP_Theo_Bayesien\tp3_data'
p=Path(dossier)
data_train=[]
data_valid=[]
fichier=list(p.glob('*.txt'))
for i in range(0,len(fichier)-1,2):
    data_train.append(fichier[i].absolute())
    data_valid.append(fichier[i+1].absolute())

# for i in range(len(data_train)):
#     print(f'------- data numero {i+1} -----')
#     frontiere=calcul_frontiere(str(data_train[i]),str(data_valid[i]))
#     taux,matrice=test(str(data_valid[i]),frontiere)
#     print(f"Le taux d'errer est {taux} %")
#     print(f"La matrice de confusion est  {matrice}")
fig,axe=plt.subplots(2,2)
k=0
for i in range(2):
    for  j in range(2):
       show2(axe,i,j,data_train[k],data_valid[k])
       k=k+1
plt.show()
# for i in range(4):
#     print(f"D1:{test(data_valid[i],calcul_frontiere(data_train[i],data_valid[i]))}")
