# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:26:52 2019

@author: katea
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
import cmath
step = 1e-3
t = np.arange(0,2+step,step)
def the_step(t):
    y = np.zeros((len(t))) #establishes array for y
    for i in range(len(t)):
        if t[i] < 0:
             y[i] = 0 
        else:
             y[i] = 1
    return y

d = 105
r = (105*math.pi)/180
#---------------------------Part 1-------------------------------------------#
step = 1e-5
t = np.arange(0,2+step,step)
def org_h(t):
    y = (0.5 + np.exp(-6*t) - (0.5*np.exp(-4*t)))
    return y

num = [1,6,12]
den = [1,10,24]
t = np.arange(0,2+step,step)
y = org_h(t) # function call using the user-defined function, shown in the 
             #above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Hand-Derived Step Response')
plt.title('Part 1')

plt.xlabel('time (s)')
plt.show()
tout2,yout2 = sig.step((num,den),T = t)
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.plot(tout2,yout2)
plt.grid(True)
plt.ylabel('Coded Step Response')
plt.title('Part 1')
plt.xlabel('time (s)')
plt.show()

num2 = [1,6,12]
den2 = [1,10,24,0]
[R,P,_]=sig.residue(num2,den2) #partial fraction expansion
print(R)
print(P)

#---------------------------Part 2-------------------------------------------#
num3 = [25250]
den3 = [1,18,218,2036,9085,25250,0]
[R,P,_] = sig.residue(num3,den3)
print(R)
print(P)

def cosine(R,P,t):
    y=np.zeros(t.shape)
    for i in range(len(R)): #goes through entire range of fraction expansion
        a = P[i].real #real part using alpha variable
        w = P[i].imag #imaginary part using p variable
        K = abs(R[i]) #absolute value of R
        ang = np.angle(R[i])
        y += (K*np.exp(a*t)*np.cos((w*t) + ang))*the_step(t) #official cosine 
        #method
    return y
    
y = 0           
t = np.arange(0,2+step,step)
y = cosine(R,P,t) # function call using the user-defined function, shown in the 
             #above cell
num4 = [25250]
den4 = [1,18,218,2036,9085,25250] #this is for coded step impulse
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Cosine Method Plot')
plt.title('Part 2')
tout3,yout3 = sig.step((num4,den4),T = t)
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.plot(tout3,yout3)
plt.grid(True)
plt.ylabel('Coded Step Impulse')
plt.title('Task 2')
plt.xlabel('time (s)')
plt.show()