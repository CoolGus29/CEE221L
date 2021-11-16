# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:19:27 2021

@author: knotm
"""

import numpy as np
import matplotlib.pyplot as plt
import os

work_path = r'C:\Users\knotm\OneDrive\Documents\Sophomore Year 2021-2022\CEE 221L'

os.chdir(work_path)
print('Current working directory %s' %work_path)


#%%
import copy
n = 1
e_appro = (1 + 1/n)**n
epsilon = np.abs(e_appro)
e_appro_list = [e_appro]
epsilon_list = [epsilon]

criterion = 1e-8
max_iter = 1e4
#e means 10 on computers
while (epsilon > criterion) & (n < max_iter):
    n += 1
    temp = copy.copy(e_appro)
    e_appro_list.append(e_appro)
    epsilon = np.abs(e_appro - temp)
    epsilon_list.append(epsilon)
    
    print('Current n is %d'%n)
    print('Current e is %f'%e_appro)
    print('Current error is %f'%epsilon)

plt.figure(figsize=(5,5))
plt.plot(np.arange(len(e_appro_list)),e_appro_list)
plt.plot(np.arange(len(e_appro_list)),np.ones_like(np.arange(len(e_appro_list)))*np.exp(1))
plt.xlabel('n')
plt.ylabel('approximated e')
plt.legend(['Approx','Truth'])
plt.show()

#%%
import copy
x = 1
y = (np.sqrt(x + 1) - np.sqrt(x))
y_list = [y]
epsilon = np.abs(y_list)
epsilon_list = [epsilon]

criterion = 1
max_iter = 1e4
while(epsilon > criterion) & (x < max_iter):
    x += 1
    temp = copy.copy(y)
    y_list.append(y)
    epsilon = np.abs(y - temp)
    epsilon_list.append(epsilon)
    
    print('current x is %d'%n)
    print('current y is %f'%y)

plt.figure(figsize=(5,5))
plt.plot(np.arange(len(y_list)),y_list)
plt.show()


#%%
x = np.arange(100)
y_1 = (np.sqrt(x + 1) - np.sqrt(x))

plt.figure(figsize=(12,6))
plt.plot(x,y_1)
plt.legend('y_1')