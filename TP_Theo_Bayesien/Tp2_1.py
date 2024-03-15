import pandas as pd
import matplotlib.pyplot as plt
from tools import  show,test
#show('tp1_data/tp1_data_train.txt')
#print(test('tp1_data/tp1_data_valid.txt',145))
"""
#2)-------Utilisation des outils sur de nouveaux jeux de donnees--------------
"""

show('tp2_data/tp2_data1_train.txt')
frontiere1=55
print(test('tp2_data/tp2_data1_valid.txt',frontiere1))
"""
#-------------------------------------------
"""
#show('tp2_data/tp2_data2_train.txt')
frontiere2=460
print(test('tp2_data/tp2_data2_valid.txt',frontiere2))
"""
#-------------------------------------------
"""
#show('tp2_data/tp2_data3_train.txt')
frontiere3=335
print(test('tp2_data/tp2_data3_valid.txt',frontiere3))
"""
"""