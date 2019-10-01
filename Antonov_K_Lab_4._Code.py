#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math


# In[3]:


plt.rcParams.update({'font.size': 14}) # set font size in plots
steps = 0.1
t = np.arange(0,20+steps,steps)
def step(t):
    y = np.zeros((len(t))) #establishes array for y
    for i in range(len(t)):
        if t[i] < 0:
             y[i] = 0 
        else:
             y[i] = 1
    return y

def ramp(t):
    y = np.zeros((len(t)))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y


# In[4]:


f = 0.25
w = 2*math.pi*f
def h1(t):
    return (np.exp(2*t)*step(1 - t))
def h2(t):
    return (step(t - 2) - step(t - 6))
def h3(t):
    return (np.cos(w*t) * step(t))


# In[6]:


t = np.arange(-10,10+steps,steps)
y = h1(t) # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('h1(t)')
plt.title('3 Functions')

y = h2(t)
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('h2(t)')

y = h3(t)
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,3)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('h3(t)')
plt.xlabel('time (s)')
plt.show()


# In[8]:


def my_conv(f1,f2):
    #from numpy import zeros
  
    Nf1 = len(f1) #define variables that are the length of both arbitrary functions
    Nf2 = len(f2)
    f1Extended = np.append(f1,np.zeros((1,Nf2 - 1))) #creates array that goes to the length of f1 - 1, 
                                                     #because arrays begin with index 0
    f2Extended = np.append(f2,np.zeros((1,Nf1 - 1)))
    result = np.zeros(f1Extended.shape) #uses shape function so that arrays are the same size
    for i in range(Nf2 + Nf1 - 2): #creates for loop that goes through range of both f1 and f2 - 2, to account
                                   #for index 0
        result[i] = 0              #creates index
        for j in range(Nf1):       #goes through first function
            if (i - j + 1 > 0):    #if length of both functions is greater than length of first function
                try:
                    result[i] = result[i] + f1Extended[j]*f2Extended[i-j+1] #adds durations of each function
                except:
                    print(i,j) #if not convolving correctly, show where it went wrong for troubleshooting
    return result
    


# In[9]:


steps = 0.1
t2 = np.arange(-20,20+steps,steps)
y = my_conv(h1(t),step(t))*steps # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f1(t) * f2(t)')
plt.title('3 Coded Convolved Functions')

#t2 = np.arange(0,20+steps,steps)
y = my_conv(h2(t),step(t))*steps # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,2)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f2(t)*f3(t)')


#t2 = np.arange(0,20+steps,steps)
y = my_conv(h3(t),step(t))*steps # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,3)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f1(t)*f3(t)')

plt.show()


# In[11]:


def org_h1(t): #these are the hand-derived functions from convolution
    return (((np.exp(2*t) * step(1 - t))/2) +((np.exp(2)*step(t-1))/2)) 
def org_h2(t):
    return ((((t-2)*step(t-2)) - (t-6)*step(t-6)))
def org_h3(t):
    return (((np.sin(w*t))/ w) * step(t))


# In[13]:


t = np.arange(-20,20+steps,steps)
y = org_h1(t) # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('f1(t) * f2(t)')
plt.title('Hand-Derived Convolutions')

y = org_h2(t)
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('f1(t) * f2(t)')

y = org_h3(t)
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,3)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('f1(t) * f2(t)')
plt.xlabel('time (s)')
plt.show()


# In[ ]:




