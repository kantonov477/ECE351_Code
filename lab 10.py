# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:30:24 2019

@author: katea
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con # this package is not included in the Anaconda
 # distribution , but should have been installed in lab 0
step = 100
R = 1e3
L = 27e-3
C = 100e-9
w = np.arange(1e3, 1e6 + step,step)
#sys = con . TransferFunction ( num , den )
#def H(jw):
    #y1 = (j*(w/R*C))/((-w**2)+(1/(L*C)) + (j*(w/(R*C))))
    #return y1
#def magn(w):
y2 = (w/(R*C))/np.sqrt((w**4 + ((1/(R*C))**2 - (2/(L*C)))*w**2 + (1/(L*C))**2))
    #return y2
#def phase(w):
y3 = (np.pi/2) - np.arctan((w/(R*C))/(-w**2 + (1/(L*C))))
    #return y3
for i in range(len(w)):
    if (-w[i]**2 + (1/(L*C))) < 0:
        y3[i] -= np.pi
#------------------Hand Derivation Plot---------------------------------------
myFigSize = (12,8)
plt.figure(figsize=myFigSize) 
plt.subplot(2,1,1) 
 
plt.semilogx(w,20*np.log10(y2)) 

plt.grid(True)
plt.ylabel('magnitude')
plt.title('Task 1 Hand Derived Bodie Plot ' )
   
plt.show()
plt.subplot(2,1,2)
plt.semilogx(w,((y3*180)/np.pi)) 

plt.grid(True)
plt.ylabel('phase angle')

plt.xlabel('f[rad]')    
plt.show()

#--------------------Code Plot-------------------------------------------------
w = np.arange(1e3, 1e6 + step,step)
num = [1/(R*C),0]
den = [1,1/(R*C),1/(L*C)]
[freq, magn2, phase2] = sig.bode((num, den))
myFigSize = (12,8)
plt.figure(2,figsize=myFigSize) 
plt.subplot(2,1,1) 
plt.xlim(1e3,1e6)
plt.semilogx(freq,magn2) 

plt.grid(True)
plt.ylabel('magnitude')
plt.title('Task 1 Code Plot ' )
   
plt.show()
plt.subplot(2,1,2)
plt.xlim(1e3,1e6)
plt.semilogx(freq,phase2) 

plt.grid(True)
plt.ylabel('phase angle')

plt.xlabel('f[rad]')    
plt.show()

#----------------------Hz Plot-------------------------------------------------
sys = con.TransferFunction ( num , den )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )
 # use _ = ... to suppress the output
 #----------------------Task 2-------------------------------------------------

fs = 1e8
step2 = 1/fs
t = np.arange(0,0.01+step2,step2)
x = np.cos(2*np.pi * 100*t) + np.cos(2*np.pi * 3024*t) + np.sin(2*np.pi * 
           50000*t)
myFigSize = (12,8)
plt.figure(3,figsize=myFigSize) 
 

plt.plot(t,x)

plt.grid(True)
plt.ylabel('output')
plt.title('Task 3 Signal ' )
   
plt.show()

[num2,den2] = sig.bilinear(num,den,fs)
y = sig.lfilter(num2,den2,x)
myFigSize = (12,8)
plt.figure(4,figsize=myFigSize) 
 

plt.plot(t,y)

plt.grid(True)
plt.ylabel('output')
plt.title('Task 3 output' )
plt.xlabel('f[Hz]')   
plt.show()
