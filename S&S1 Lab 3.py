#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# In[2]:


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


# In[3]:


t = np.arange(0,20+steps,steps)
def f1(t):
    
    return (step(t-2) - step(t-9)) #functions included in the lab handout
 

def f2(t):
    
    return (np.exp(-t))
   

def f3(t): 
    
    return ((ramp(t - 2)*(step(t - 2) - step(t - 3))) + (ramp(4-t)*(step(t - 3) - step(t - 4))))
    


# In[4]:


t = np.arange(0,20+steps,steps)
y = f1(t) # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 1')
plt.title('3 Functions')

y = f2(t)
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 2')

y = f3(t)
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,3)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('Function 3')
plt.xlabel('time (s)')
plt.show()


# In[5]:


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
    
   


# In[6]:


steps = 0.1
t2 = np.arange(0,40+steps,steps)
y = my_conv(f1(t),f2(t))*steps # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f1(t) * f2(t)')
plt.title('3 Functions')

#t2 = np.arange(0,20+steps,steps)
y = my_conv(f2(t),f3(t))*steps # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,2)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f2(t)*f3(t)')
plt.title('3 Functions')

#t2 = np.arange(0,20+steps,steps)
y = my_conv(f1(t),f3(t))*steps # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,3)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f1(t)*f3(t)')
plt.title('3 Functions')
plt.show()


# In[7]:


t2 = np.arange(0,40+steps,steps)
y = sig.convolve(f1(t),f2(t)) # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,1)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f1(t) * f2(t)')
plt.title('3 Functions')

#t2 = np.arange(0,40+steps,steps)
y = sig.convolve(f2(t),f3(t)) # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,2)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f2(t)*f3(t)')
plt.title('3 Functions')

#t2 = np.arange(0,40+steps,steps)
y = sig.convolve(f1(t),f3(t)) # function call using the user-defined function, shown in the above cell
myFigSize = (10,10)
plt.figure(figsize=myFigSize)
plt.subplot(3,1,3)
plt.plot(t2,y)
plt.grid(True)
plt.ylabel('f1(t)*f3(t)')
plt.title('3 Functions')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




