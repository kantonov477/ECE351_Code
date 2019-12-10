# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:26:54 2019

@author: katea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fftpack import fft, fftshift
import control as con # this package is not included in the Anaconda
 # distribution , but should have been installed in lab 0
fs = 1e6
# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df ['0'].values
sensor_sig = df ['1'].values

plt . figure ( figsize = (10 , 7) )
plt . plot (t , sensor_sig )
plt . grid ()
plt . title ('Noisy Input Signal')
plt . xlabel ('Time [s]')
plt . ylabel ('Amplitude [V]')
plt . show ()



def make_stem ( ax ,x ,y , color ='k', style ='solid', label ='', 
               linewidths =2.5 ,** kwargs ) :
    ax . axhline ( x [0] , x [ -1] ,0 , color ='r')
    ax . vlines (x , 0 ,y , color = color , linestyles = style , 
                 label = label , linewidths =
                 linewidths )
    ax . set_ylim ([1.05* y . min () , 1.05* y . max () ])
#----------------------------Part 1--------------------------------------------
def clean_FFT(x,fs):
    N = len( x ) # find the length of the signal
    X_fft = fft ( x ) # perform the fast Fourier transform (fft)
    X_fft_shifted = fftshift ( X_fft ) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange ( - N /2 , N /2) * fs / N # compute the frequencies for 
    #the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    X_phi = np.angle ( X_fft_shifted )
    X_mag = np.abs( X_fft_shifted ) / N # compute the magnitudes of the signal
    for i in range (len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0
    return X_mag, X_phi, freq#so that no significantly small magnitude values 
            #will be accounted
     # compute the phases of the signal
    # ----- End of user defined function ----- #
step = 1e1
frequency = np.arange(1e1, 1e5+step,step)
w = 2*np.pi*frequency #This is for Part 3 bode plots
#This runs the signal through FFT with different ranges  
X_mag, X_phi, freq = clean_FFT(sensor_sig,fs)      
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq, X_mag )
plt.xscale('log')
plt.xlim(1e0,100e3)
plt.title('Task 1 FFT Total Range from 1 to 100000 Hz' )
plt.ylabel('sensor signal FFT output')
plt.xlabel('f[Hz]')
plt . show ()

X_mag, X_phi, freq = clean_FFT(sensor_sig,fs)      
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq, X_mag )
plt.xscale('log')
plt.xlim(1e0,1.8e3)
plt.title('Task 1 FFT Range from 1 to 1800 Hz' )
plt.ylabel('sensor signal FFT output')
plt.xlabel('f[Hz]')
plt . show ()  

X_mag, X_phi, freq = clean_FFT(sensor_sig,fs)      
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq, X_mag )
plt.xscale('log')
plt.xlim(1.8e3,2e3)
plt.title('Task 1 FFT Range from 1.8 to 2kHz' )
plt.ylabel('sensor signal FFT output')
plt.xlabel('f[Hz]')
plt . show () 

X_mag, X_phi, freq = clean_FFT(sensor_sig,fs)      
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq, X_mag )
plt.xscale('log')
plt.xlim(2e3,100e3)
plt.title('Task 1 FFT Range from 2 to 100kHz' )
plt.ylabel('sensor signal FFT output')
plt.xlabel('f[Hz]')
plt . show ()  
#----------------------Part 3--------------------------------------------------
L = 70.16e-3 #H
R = 4.95e2 #ohms
C = 0.1e-6 #F
num = [1/(R*C),0]
den = [1,1/(R*C),1/(L*C)] #this is the transfer function in numerator and 
#denominator
[freq, magn, phase] = sig.bode((num, den))
#This plots the bode plots
#----------------------Hz Plot-------------------------------------------------
myFigSize = (12,8)
plt.figure(figsize=myFigSize)

sys = con.TransferFunction ( num , den )
plt.title("Task 3 Full Range Bode Plot") 
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

 # use _ = ... to suppress the output
myFigSize = (12,8)
plt.figure(figsize=myFigSize)

sys = con.TransferFunction ( num , den )
_ = con.bode ( sys , np.arange(1e1, 1.8e3+step,step)*2*np.pi , dB = True ,
              Hz = True , deg = True , Plot = True )
plt.title("Task 3 0 to 1.8 kHz Bode Plot") 
myFigSize = (12,8)
plt.figure(figsize=myFigSize)
 
sys = con.TransferFunction ( num , den )
_ = con.bode ( sys , np.arange(1.8e3, 2e3+step,step)*2*np.pi  , dB = True , 
              Hz = True , deg = True , Plot = True )
plt.title("Task 3 1.8 to 2 kHz Bode Plot")
myFigSize = (12,8)
plt.figure(figsize=myFigSize)

sys = con.TransferFunction ( num , den )
_ = con.bode ( sys , np.arange(2e3, 100e3+step,step)*2*np.pi  , dB = True , 
              Hz = True , deg = True , Plot = True )
plt.title("Task 3 2 to 100 kHz Bode Plot") 
myFigSize = (12,8)
plt.figure(figsize=myFigSize)

sys = con.TransferFunction ( num , den )
_ = con.bode ( sys , np.arange(1e6, 1e8+step,step)*2*np.pi  , dB = True , 
              Hz = True , deg = True , Plot = True )
plt.title("Task 3 1e6 to 1e8 Hz Bode Plot") 
#----------------------------Part 4--------------------------------------------
[num2,den2] = sig.bilinear(num,den,fs) #translates to z domain
y = sig.lfilter(num2,den2,sensor_sig) #filters signal with band-pass design
myFigSize = (12,8)
plt.figure(figsize=myFigSize) 
 

plt.plot(t,y)

plt.grid(True)
plt.ylabel('output')
plt.title('Task 4 Filtered Signal Output' )
plt.xlabel('f[Hz]')   
plt.show()
#runs FFT with filtered signal now
X_mag2, X_phi2, freq2 = clean_FFT(y,fs)      
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq2, X_mag2 )
plt.xscale('log')
plt.xlim(1e0,100e3)
plt.ylabel('filtered signal FFT output')
plt.xlabel('f[Hz]')
plt.title('Task 4 FFT Total Range from 1 to 100000 Hz' )
plt . show ()

X_mag2, X_phi2, freq2 = clean_FFT(y,fs)    
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq2, X_mag2 )
plt.xscale('log')
plt.xlim(1e0,1.8e3)
plt.ylabel('filtered signal FFT output')
plt.xlabel('f[Hz]')
plt.title('Task 4 FFT Range from 1 to 1800 Hz' )
plt . show ()  

X_mag2, X_phi2, freq2 = clean_FFT(y,fs)     
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq2, X_mag2 )
plt.xscale('log')
plt.xlim(1.8e3,2e3)
plt.ylabel('filtered signal FFT output')
plt.xlabel('f[Hz]')
plt.title('Task 4 FFT Range from 1.8 to 2 kHz ' )
plt . show () 

X_mag2, X_phi2, freq2 = clean_FFT(y,fs)      
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq2, X_mag2 )
plt.xscale('log')
plt.xlim(2e3,100e3)
plt.ylabel('filtered signal FFT output')
plt.xlabel('f[Hz]')
plt.title('Task 4 FFT Range from 2 to 100 kHz' )
plt . show ()  