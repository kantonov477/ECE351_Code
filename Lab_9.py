# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:49:11 2019

@author: katea
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
fs = 100
step = 1/fs
T = 8 
t = np.arange(0,2,step)
x = [np.cos(2*np.pi*t), 5*np.sin(2*np.pi*t), 2*np.cos((2*np.pi * 2*t) - 2) + 
     (np.sin((2*np.pi*6*t) + 3))] #made array of x so i could run them through 
#a for loop
def FFT(x,fs):
    N = len( x ) # find the length of the signal
    X_fft = fft ( x ) # perform the fast Fourier transform (fft)
    X_fft_shifted = fftshift ( X_fft ) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange ( - N /2 , N /2) * fs / N # compute the frequencies for 
    #the 
    #output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    X_mag = np.abs( X_fft_shifted ) / N # compute the magnitudes of the signal
    X_phi = np.angle ( X_fft_shifted ) # compute the phases of the signal
    # ----- End of user defined function ----- #
    return X_mag, X_phi, freq

for h in [1,2,3]:
        [X_mag, X_phi, freq] = FFT(x[h-1],fs)
        
        myFigSize = (12,8)
        plt.figure(h, figsize=myFigSize)    
        plt.subplot(3,1,1)
        plt.plot(t,x[h-1])
        plt.grid(True)
        plt.ylabel('x(t)')
        plt.title('Task 1 - User Defined FFT of x(t)' ) #graph of actual 
        #function
        plt.xlabel('t (s)')    
        plt.show()
        
        plt.subplot(3,2,3)
        
        plt.stem ( freq, X_mag, use_line_collection=True )
        plt.grid(True)
        plt.ylabel('X_magn') #magnitude, not zoomed in
        #plt.xlabel('frequency')    
        plt.show()
        
        plt.subplot(3,2,4)
        plt.stem ( freq , X_mag,use_line_collection=True)
        plt.grid(True)
        if h == 3: #magnitude, zoomed in
            plt.xlim(-15,15)
        else:
            plt.xlim(-2,2)
        
        #plt.ylabel('X_magn')
        #plt.xlabel('frequency')    
        plt.show()
        
        plt.subplot(3,2,5)
        plt.stem ( freq , X_phi,use_line_collection=True )
        plt.grid(True)
        
        plt.ylabel('X_phi') #phase angle, not zoomed in
        plt.xlabel('frequency')    
        plt.show()
        
        plt.subplot(3,2,6)
        plt.stem ( freq, X_phi,use_line_collection=True )
        plt.grid(True)
        if h == 3: 
            plt.xlim(-15,15)
        else:
            plt.xlim(-2,2)
        
        
        #plt.ylabel('X_magn')
        plt.xlabel('frequency')   #phase angle zoomed in 
        plt.show()  
        
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
            X_phi[i] = 0 #so that no significantly small magnitude values 
            #will be accounted
     # compute the phases of the signal
    # ----- End of user defined function ----- #

    
    return X_mag, X_phi, freq

for h in [1, 2, 3]:
    [X_mag, X_phi, freq] = clean_FFT(x[h-1],fs)
        
           
    myFigSize = (12,8)
    plt.figure(h+3, figsize=myFigSize)    
    plt.subplot(3,1,1)
    plt.plot(t,x[h-1])
    plt.grid(True)
    plt.ylabel('x(t)')
    plt.title('Task 2  User Defined clean_FFT of x(t)' )
    plt.xlabel('t (s)')    
    plt.show()
    
    plt.subplot(3,2,3)
        
    plt.stem ( freq, X_mag, use_line_collection=True )
    plt.grid(True)
    plt.ylabel('X_magn')
    #plt.xlabel('frequency')    
    plt.show()
        
    plt.subplot(3,2,4)
    plt.stem ( freq , X_mag,use_line_collection=True)
    plt.grid(True)
    if h == 3: 
        plt.xlim(-15,15)
    else:
        plt.xlim(-2,2)
        
    #plt.ylabel('X_magn')
    #plt.xlabel('frequency')    
    plt.show()
        
    plt.subplot(3,2,5)
    plt.stem ( freq , X_phi,use_line_collection=True )
    plt.grid(True)
        
    plt.ylabel('X_phi')
    plt.xlabel('frequency')    
    plt.show()
        
    plt.subplot(3,2,6)
    plt.stem ( freq, X_phi,use_line_collection=True )
    plt.grid(True)
    if h == 3: 
        plt.xlim(-15,15)
    else:
        plt.xlim(-2,2)
            
    
        
        
    #plt.ylabel('X_magn')
    plt.xlabel('frequency')    
    plt.show()  
 

fs = 100
step = 1/fs
T = 8
N = 15
t = np.arange(0, 2*T, step)

y = 0

for k in range(1,N+1): #goes through entire range of N
    #
    b = (2/(k*np.pi))*(1 - np.cos(k*np.pi))
    x = b*np.sin(k*((2*np.pi)/T)*t) #official fourier series equation
    y = y + x #add both b constant and x
# For every k up to the max we need, add the contributions at 
# that frequency
t2 = np.arange(0,2, step)
[X_mag, X_phi,freq] = clean_FFT(y, fs)
myFigSize = (12,8)
plt.figure(7, figsize=myFigSize)    
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('x(t)')
plt.title('Task 3 - Series Approximation with clean-FFT of x(t) ' )
plt.xlabel('t (s)')    
plt.show()
    
plt.subplot(3,2,3)
    
plt.stem ( freq, X_mag, use_line_collection=True )
plt.grid(True)
plt.ylabel('X_magn')
#plt.xlim(-40,40)  
plt.show()
    
plt.subplot(3,2,4)
plt.stem ( freq , X_mag,use_line_collection=True)
plt.grid(True)

plt.xlim(-2,2)
    
 
plt.show()
    
plt.subplot(3,2,5)
plt.stem ( freq , X_phi,use_line_collection=True )
plt.grid(True)
    
plt.ylabel('X_phi')
plt.xlabel('frequency')  
#plt.xlim(-40,40)   
plt.show()
    
plt.subplot(3,2,6)
plt.stem ( freq, X_phi,use_line_collection=True )
plt.grid(True)

plt.xlim(-2,2)
        

plt.xlabel('frequency')    
plt.show()  


           

   
                