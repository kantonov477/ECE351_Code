# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
step = 1e-5
t = np.arange(0,1.2e-3+step,step)
def the_step(t):
    y = np.zeros((len(t))) #establishes array for y
    for i in range(len(t)):
        if t[i] < 0:
             y[i] = 0 
        else:
             y[i] = 1
    return y
step = 1e-5
t = np.arange(0,1.2e-3+step,step)
R = 1000
L = 0.027
C = 100e-9
num = [1/(R*C),0]
den = [1,1/(R*C),1/(L*C)]
tout,yout = sig.impulse((num,den),T = t)

d = 105
r = (105*math.pi)/180
step = 1e-5
t = np.arange(0,1.2e-3+step,step)
def org_h(t):
    y = (10356)*np.exp(-5000*t)*np.sin((18584*t) + r)*the_step(t)
    return y

#step = 1e-5
#t = np.arange(0,1.2e-3+step,step)
y = org_h(t) # function call using the user-defined function, shown in the 
             #above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Hand-Derived Impulse Function')
plt.title('Task 1')

myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(2,1,2)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('Coded Impulse Function')

plt.xlabel('time (s)')
plt.show()

tout2,yout2 = sig.step((num,den),T = t)
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.plot(tout2,yout2)
plt.grid(True)
plt.ylabel('Coded Step Impulse')
plt.title('Task 2')
plt.xlabel('time (s)')
plt.show()
