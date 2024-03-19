import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('tp4_data/tp4_data1_train.txt',names=['x','y'])
# plt.figure(figsize=(12,4))
# plt.hist(df[df['y']==0]['x'],bins=25,color='red',label='Classe 0',alpha=0.7)
# plt.hist(df[df['y']==1]['x'],bins=25,color='blue',label='Classe 1',alpha=0.7)
# plt.hist(df[df['y']==2]['x'],bins=25,color='green',label='Classe 2',alpha=0.7)
# plt.axvline(142,c='yellow')
# plt.axvline(180,c='yellow')
# plt.legend()
# plt.show() 
# plt.title(f"frontier142 frontier2 180")
frontiere1=142
frontiere2=180
df_valid=pd.read_csv('tp4_data/tp4_data1_valid.txt',names=['x','y'])
X_valid=np.array(df_valid['x'])
ypredi=[]
for i in X_valid:
    if i<142:
        ypredi.append(0)
    elif i>142 and i <180:
        ypredi.append(1)
    else:
        ypredi.append(2)
y_valid=np.array(df_valid['y'])
errr=sum([1 for i in range(0,len(y_valid)) if y_valid[i]!=ypredi[i]])
Taux=errr/len(ypredi)*100
print(f"Taux d'errer {Taux} %")
positif_positif=len([0 for i in range(len(ypredi)) if ypredi[i]==0 and y_valid[i]==0])
negatif_negatif=len([0 for i in range(len(ypredi)) if ypredi[i]==1 and y_valid[i]==1])
positif_negatif=len([0 for i in range(len(ypredi)) if ypredi[i]==0 and y_valid[i]==1])
negatif_positif=len([0 for i in range(len(ypredi)) if ypredi[i]==1 and y_valid[i]==0])
mat_conf=np.array([[positif_positif,positif_negatif],[negatif_positif,negatif_negatif]])
print(mat_conf)