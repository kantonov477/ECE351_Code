#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) # set font size in plots
steps = 1e-2
t = np.arange(0,10+steps,steps) # to go up to 5.0, we must add a stepSize since `np.arange()`
 # goes up to (without including) the value of the second argument
print('# of elements: len(t) =',len(t), # notice this may be one larger than expected since `t` starts at 0
 '\nFirst element: t[0] =',t[0], # index the first value of the array `t`
 '\nLast element: t[len(t)-1] =',t[len(t)-1]) # index the last value of the array `t`
# Note that despite `t` having a length of 501, we index the 500th element since the first element is 0


# In[24]:


# --- User-Defined Function --- #
# Create the output `y(t)` using a for loop and if/else statements
def func1(t):
    y = np.zeros((len(t),1)) # initialize `y` as a numpy array (of zeros)

    for i in range(len(t)):
        #if i < (len(t)+1) / 3:
             #y[i] = t[i]**2
        #else:
        y[i] = np.cos(t[i])
    return y


# In[25]:


y = func1(t) # function call using the user-defined function, shown in the above cell
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('y(t) = cos(t) with Good Resolution')
plt.title('Background - Illustration of for Loops and if/else Statements')
plt.show()


# In[42]:


def step(t):
    y = np.zeros((len(t),1)) #establishes array for y
    for i in range(len(t)):
        if t[i] < 0:
             y[i] = 0 
        else:
             y[i] = 1
    return y

def ramp(t):
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y
y = step(t)
steps = 1e-2
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('u(t)')
plt.title('Step Function')
plt.show()

y = ramp(t)
steps = 1e-2
myFigSize = (10,9)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('r(t)')
plt.title('Ramp Function')
plt.show()


      


# In[27]:


def my_step(t):
    return (ramp(t-0) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6)) #equation for whole plotted graph


# In[28]:


steps = 1e-2
t = np.arange(-5,10+steps,steps) 

y = my_step(t)
myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.show()


# In[36]:


steps = 1e-2
t = np.arange(-10,20+steps,steps) 

y = my_step(-t) #time reversal
myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.show()
    


# In[37]:


steps = 1e-2
t = np.arange(-20,20+steps,steps) 

y = my_step(t-4)
myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.show()


# In[32]:


steps = 1e-2
t = np.arange(-20,20+steps,steps) 

y = my_step(-t-4)
myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.show()


# In[39]:


steps = 1e-2
t = np.arange(-5,20+steps,steps) 

y = my_step(t/2)
myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.subplot(1,1,1)
plt.plot(t,y)
plt.grid(True)
plt.show()


# In[52]:


import numpy as np
steps = 1e-2
t = np.arange(-5,10+steps,steps) 
y = my_step(t)
dt = np.diff(t) #had to define dt first
dy = np.diff(y, axis=0) / dt #with dt defined, dy could be defined using the actual equation

myFigSize = (10,8)
plt.figure(figsize=myFigSize)
plt.plot(t,y,'--', label='y(t)')
plt.plot(t[range(len(dy))], dy[:,0], label = 'dy(t)/dt')
plt.xlabel('t')
plt.ylabel('y(t), dy(t)/dt')
plt.title('Differentiation wrt Time')
plt.legend()
#plt.subplot(1,1,1)
#plt.plot(t,dy/dt)
plt.grid(True)
plt.ylim([-2, 10])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




