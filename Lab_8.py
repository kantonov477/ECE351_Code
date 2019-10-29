# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:28:27 2019

@author: katea
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
#-------------------------------Part 1-----------------------------------------
def ak(k):
    a = 0 #because the function is odd
    return a

def bk(k):
    b = (2/(k*np.pi))*(1 - np.cos(k*np.pi)) #this part was calculated in the
    #pre-lab
    return b

a0 = ak(0)
print("a_0 = ",a0) #non-zeros values of the fourier series were printed
a1 = ak(1)
print("a_1 = ",a1)
b1 = bk(1)
print("b_1 = ",b1)
b2 = bk(2)
print("b_2 = ",b2)
b3 = bk(3)
print("b_3 = ",b3)

#-----------------------------Part 2-------------------------------------------
step = 1e-3
t = np.arange(0,20+step,step)
T = 8 #period

N = [1,3,15,50,150,1500] #values we want to plot Fourier series approximations
y = 0 #initialize variable

for h in [1,2]:#so that we can have two figures only
    for i in (1 + (h-1)*3,2 + (h-1)*3,3 + (h-1)*3): #so that we can have 3 
        #plots per figure
        for k in np.arange(1,N[i - 1] + 1): #goes through entire range of N
            b = (2/(k*np.pi))*(1 - np.cos(k*np.pi))
            x = b*np.sin(k*((2*np.pi)/T)*t) #official fourier series equation
            y = y + x #add both b constant and x
        myFigSize = (12,8)
        plt.figure(h, figsize=myFigSize)    
        plt.subplot(3,1, i - (h - 1)*3) #so that N value and plot aligns
        plt.plot(t,y)
        plt.grid(True)
        plt.ylabel('N = %i' %N[i - 1]) #so that correct N value is labeled
        if i == 1 or i == 4: #these are the top of each figure
            plt.title('Fourier Series Approximation')
        if i == 3 or i == 6:
            plt.xlabel('t (s)')    
            plt.show()       
        y = 0
   
            