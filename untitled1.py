# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:46:22 2021

@author: knotm
"""

import numpy as np
import matplotlib.pyplot as plt
import os

work_path = r'C:\Users\knotm\OneDrive\Documents\Sophomore Year 2021-2022\CEE 221L'

os.chdir(work_path)
print('Current working directory %s' %work_path)

#%%
x_size = 10
y_size = 50
image = np.zeros((y_size,x_size))
#image = np.zeros((100,100))

x = np.arange(100)
#y = (np.sin(x/100*2*np.pi)*25).astype(int) + 50
y = (x**2).astype(int)

plt.figure(figsize=(12,6))
for i,y_pixel in enumerate(y):
    if (y_pixel < y_size) & (x[i] < x_size):
        image[y_pixel,x[i]] = 1
plt.imshow(image)
plt.colorbar()
#image[y,x] = 1
#%% Visualization
plt.figure(figsize=(12,6))
plt.imshow(image)
plt.colorbar()

#%%
x = np.arange(100)
y = (np.sin(x/100*2*np.pi)*25).astype(int) + 50
plt.figure(figsize=(12,6))
plt.scatter(x,y)
plt.plot(x,np.sin(x/100*2*np.pi)*25+50,'r')
plt.legend(['curve','dots'])

#%% visualization step
x = np.arange(100)
y = (np.sin(x/100*2*np.pi)*25) + 50

plt.figure(figsize=(12,6))
plt.scatter(x,y)
plt.plot(x,y,'r')
plt.legend(['curve','dots'])

#%% will have to reproduce some verison of this for homework
x = np.arange(100)
y_1 = x + 1
y_2 = x**2 + x +1
y_3 = x**3 + x**2 + x +1

plt.figure(figsize=(12,6))
plt.plot(x,y_1,'r-.')
plt.plot(x,y_2,'b-')
plt.plot(x,y_3,'b--')
plt.legend(['y_1','y_2','y_3'])

plt.figure(figsize=(12,6))
plt.plot(x,np.log(y_1),'r-.')
plt.plot(x,np.log(y_2),'b-')
plt.plot(x,np.log(y_3),'b--')
plt.legend(['log(y_1)','log(y_2)','log(y_3)'])

#%%
a = 50
x = np.linspace(0,2*a,50)
# y = a ** x
y_sin = np.sin(x/a*np.pi)
y_cos = np.cos(x/a*np.pi)

plt.figure(figsize=(5,5))
plt.plot(x,y_sin, 'r', label='sin curve')
plt.plot(x,y_cos, 'b', label='cos curve')
plt.show()
plt.legend()


#%% pg39 Python along with its standard library has a s
#numerical data, strings, boolean(true or false) values,
#'single value' types are sometimes called scalar types
ival = 17239871

fval = 7.243

fval2 = 6.78e-5

#integer division not resulting  in a whole number will always
val = 5/3
type(val)

val = np.int(5/3)
type(val)

#to get C-style integer division which drops the fraction
#whole number, use the floor division operator //
val = 5//3
type(val)


#%%string values normally used for path operations
a = r'C:\Users\knotm\OneDrive\Documents\Sophomore Year 2021-2022\CEE 221L'

#%% Control flow If
x = -2

if x < 0:
    print ('It\'s negative')
elif x < -1:
    print('Equal to zero')
elif 0 < x < 5:
    print('Positive but smaller than 5')
else:
    print('Positive and larger than or equal to 5')

#%% Control flow for loops
sequence = [1, 2, None, 4, None, 5]
total = 0
for i,value in enumerate(sequence):
    print('Checking the %d-th item' %i)
    if value is None:
        print('The item is None')
    else:
        total == value
        print('Current total is %d' %total)

#%%
sequence = [1, 2, None, 4, None, 5]
total = 0
for value in sequence:
    if value is None:
        continue #jump to the next iteration
    total == value
print ('Current total is %d' %total)

#%% Shared memory and copy experiment
a = [1,2,3,4,]
#b = a
b = a.copy()
a.append(5)
print('b =', b)





#%% generate Fibonacci series and compute the fibonacci

series_length = np.int(input("Please enter a number:\n"))
fibonacci = np.empty(series_length)

for i in np.arange(series_length):
    if i == 0:
        fibonacci[i] = 0
    elif i == 1:
        fibonacci[i] = 1
    else:
        fibonacci[i] = fibonacci[i-1] + fibonacci[i-2]
        
print(fibonacci)

plt.figure(figsize=(5,6))
plt.subplot(211)
plt.plot(np.arange(series_length), fibonacci, 'ro')
plt.title('fibonacci series')
plt.ylabel('fibonacci number')
plt.xlabel('index')
plt.subplot(212)
plt.plot(np.arange(series_length-2)+1, fibonacci[2:]/fibonacci[1:-1])
plt.ylabel('ratio')
plt.xlabel('index')
plt.title('fibonacci ratio')
plt.subplots_adjust(hspace=0.5)
plt.show()

#%%operations on list
#adding and removing elements

my_list = ['foo', 'peekaboo', 'baz', 'dwarf']

my_list.append(6)

my_list.remove('foo')

#concatenating and extending
x = [4, None, 'foo']
y = [7, 8, (2, 3)]

z = x + y

x = x.extend(y)

#%%sequence funtions
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two']


for i,val in enumerate(seq1):
    print(i)
    print(val)

i = [1,2,3]
for name1,name2,idx in zip(seq1,seq2,i):
    print((idx,name1,name2))
    
a = np.arange(10)
b = np.arange(10)+2
c = a * b

    
    
    
#%% dictionary





#%% practice while-loop (use for the homework)
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


#%% practice of for-loop
N_total = 1000
my_pi = np.zeros(N_total)

for i,N in enumerate(np.arange(N_total)+1):
    x = np.random.rand(N) - 0.5
    y = np.random.rand(N) - 0.5
    flag = np.empty_like(x)
    flag[:] = np.nan
    r = np.sqrt(x**2+y**2)
    flag = r<=0.5
    ratio = np.sum(flag)/len(flag)
    my_pi[i] = 4*ratio
#print(my_pi)

plt.figure(figsize=(5,5))
plt.plot(np.arange(N_total)+1, my_pi)
plt.plot(np.arange(N_total)+1, np.ones_like(np.arange(N_total)+1)*np.pi)
plt.xlabel('No. of sample')
plt.ylabel('approximated pi')
plt.show()


#%%3D Visualization
res = 5000 #res means resolution

x = np.linspace(-10,10,res)
y = np.linspace(-10,10,res)

X,Y = np.meshgrid(x,y)

Z = X**2 + Y**2

def my_fun(X,Y):
    

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

plt.show()

Z_mean = np.sum(Z)/res**2

Area = 4
V_1 = Area * Z_mean

ele_area = (2/res)**2

V_2 = ele_area * np.sum(Z)

print('V_1=', V_1)
print('V_2=', V_2)
#%%matrix concatenation
# np.concatenate()

X = np.random.rand(3,3) 
Y = np.random.rand(3,3)

M = np.concatenate((X,Y),axis=0)
M = np.stack((X,Y),axis=0)

#%% read in data from file and perform linear regression
#Using excel this can also be done, but it is much faster and easier in python
file = r'C:\Users\knotm\OneDrive\Documents\Sophomore Year 2021-2022\CEE 221L\NGES_data.csv'
data = np.genfromtxt(file, dtype=float, delimiter = ",", skip_header=1)

x = data[:,0]
y = data[:,1]
            #or 2
A = np.vstack([np.ones(len(x)), x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10]).T

# k, b = np.linalg.lstsq(A, y, rcond=None)[0]
coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
#we always use the np.linalg.lstsq to maintain it as a linear model no matter the level

import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Original', markersize=10)
plt.plot(x, A @ coefficients[:,np.newaxis], 'r', label='Fitted Line')
plt.xlabel('Depth[m]')
plt.ylabel('Qt[-]')
plt.legend()

#%%
def my_product(v1,v2):
    """
    arguments:
        x:float np.array:1st input
        y:float np.array:2nd input
    return:
        z:float np.array:result
    """
    return v1 * v2


x = np.arange(5)
y = np.arange(5)+1

z = my_product(v1=x,v2=y)

#%%

def linear_regression(x,y,order):
    """
    arguments:
        x:float np.array:x vector
        y:float np.array:y vector
        order:int:highest order
    return:
        coefficients: float np.array:the coefficients of all terms
        
    """
    A = np.ones(len(x))
    for i in np.arange(order) + 1:
        A = np.vstack([A, x**i])
    A = A.T
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    return coefficients, A

file = r'C:\Users\knotm\OneDrive\Documents\Sophomore Year 2021-2022\CEE 221L\NGES_data.csv'
data = np.genfromtxt(file, dtype=float, delimiter = ",", skip_header=1)

x = data[:,0]
y = data[:,1]

coefficients, A = linear_regression(x,y,order=11)

plt.plot(x, y, 'o', label='Original', markersize=10)
plt.plot(x, A @ coefficients[:,np.newaxis], 'r', label='Fitted Line')
plt.xlabel('Depth[m]')
plt.ylabel('Qt[-]')
plt.legend()

#%%
import my_module
print (my_module.CONST)

x = np.arange(5)
y = np.arange(5) + 1

z = my_module.my_product(x,y)