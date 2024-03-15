import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('tp4_data/tp4_data1_train.txt',names=['x','y'])
plt.figure(figsize=(12,4))
plt.hist(df[df['y']==0]['x'],bins=25,color='red',label='Classe 0',alpha=0.7)
plt.hist(df[df['y']==1]['x'],bins=25,color='blue',label='Classe 1',alpha=0.7)
plt.hist(df[df['y']==2]['x'],bins=25,color='green',label='Classe 2',alpha=0.7)
plt.legend()
plt.show()
