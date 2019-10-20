# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:23:22 2019

@author: katea
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#--------------------Part 1---------------------------------------------------
num1 = [1,9] #G(s) transfer function
den1 = [1,-2,-40,-64]
[Z1,P1,K1] = sig.tf2zpk(num1, den1) #identifies poles and zeros

print('zeroes of G(s) = ',Z1)
print('poles of G(s) = ',P1)
print('gains of G(s) = ',K1)
#----------A(s)--------------------
num2 = [1,4] #A(s) transfer function
den2 = [1,4,3]
[Z2,P2,K2] = sig.tf2zpk(num2, den2)
print('zeroes of A(s) = ',Z2)
print('poles of A(s) = ',P2)
print('gains of A(s) = ',K2)
#----------B(s)--------------------
num3 = [1,26,168] #B(s) transfer function

[Z3,Z4] = np.roots(num3) # there is no denominator, must use roots function
print('zeroes of B(s) = ',Z3, Z4)

#-----------------Convolving-------------------------------
step = 1e-3
t = np.arange(0,10+step,step)
num4 = [1,9]
den4 = sig.convolve([1,3],[1,1]) #open loop equation is A*G denominator
#convolution
den5 = sig.convolve([1,-8],[1,2])  #did this in two parts
den6 = sig.convolve(den4, den5)
tout,yout = sig.step((num4,den6),T = t)
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('y(t) ')
plt.title('Open Loop Transfer Function')
plt.xlabel('time (s)')
plt.show()

#--------------------Part 2---------------------------------------------
numA = [1,4]
denA = [1,4,3]
numG = [1,9]
denG = [1,-2,-40,-64] 
#closed loop equation is (numA*numG)/((den G + numBnumG)denA)
numB = [1,26,168]
numCL = sig.convolve(numA,numG)
denCL = sig.convolve(denG + sig.convolve(numB,numG), denA)
[ZCL,PCL,KCL] = sig.tf2zpk(numCL, denCL)

print('zeroes of Closed Loop = ',ZCL)
print('poles of Closed Loop = ',PCL)
print('gains of Closed Loop = ',KCL)
step = 1e-3
t = np.arange(0,10+step,step)
tout,yout = sig.step((numCL,denCL),T = t)
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Closed Loop Transfer Function')
plt.xlabel('time (s)')
plt.show()

