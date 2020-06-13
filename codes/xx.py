#!/usr/bin/env python
# coding: utf-8

# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import numpy as np
import pandas as pd
print(plt.style.available)


# In[182]:


print(np.finfo(np.float32))
print(np.finfo(np.float64))
print(np.finfo(np.longdouble))
print(np.float32(1/3))
print(np.float64(1/3))
print(np.longdouble(1.0)/3)


# # $x^3$ error plot

# ## First order 

# In[185]:


max_val = 12
dx = 0.1**np.arange(0.1, max_val,0.1)
error = np.zeros(len(dx))
error = -((1+dx)**3-1.0)/dx + 3.0
log_errors = np.log10(np.abs(error))


mdx = 0.1**np.arange(0.1, max_val,0.1,dtype=np.longdouble)
merror = np.zeros(len(mdx),dtype=np.longdouble)
merror = -((np.float128(1)+mdx)**3-np.float128(1.0))/mdx + np.float128(3.0)
mlog_errors = np.log10(np.abs(merror))

ldx = 0.1**np.arange(0.1, max_val,0.1,dtype='f')
lerror = np.zeros(len(ldx),dtype='f')
lerror = -((1+ldx)**3-1.0)/ldx + 3.0
llog_errors = np.log10(np.abs(lerror))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(np.abs(ldx)), llog_errors,dashes=[1, 2],label="float32")
ax.plot(np.log10(np.abs(dx)), log_errors,label="float64")
ax.plot(np.log10(np.abs(mdx)), mlog_errors,dashes=[3, 2],label="float128")
ax.set_xlim(-0.1, -max_val)  # decreasing time
ax.set_xlabel('log(step size)')

ax.set_ylabel('log(error)')
ax.set_title('Error in first order finite difference approximation of the derivative of $x^3$')

ax.legend()
# ax.grid(True)
fig.savefig("/home/himanshu/Desktop/master_project/thesis/images/x^3_error_order1.png",bbox_inches='tight')


# ## Second order

# In[159]:



max_val = 8
dx = 0.1**np.arange(0.1, max_val,0.1)
error = np.zeros(len(dx))
error = -((1+dx)**3-(1-dx)**3)/(2*dx) + 3.0
log_errors = np.log10(np.abs(error))


mdx = 0.1**np.arange(0.1, max_val,0.1,dtype=np.longdouble)
merror = np.zeros(len(mdx),dtype=np.longdouble)
merror = -((1+mdx)**3-(1-mdx)**3)/(2*mdx) + 3.0
mlog_errors = np.log10(np.abs(merror))


ldx = 0.1**np.arange(0.1, max_val,0.1,dtype='f')
lerror = np.zeros(len(ldx),dtype='f')
lerror = -((1+ldx)**3-(1-ldx)**3)/(2*ldx) + 3.0
llog_errors = np.log10(np.abs(lerror))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(np.abs(ldx)), llog_errors,dashes=[1, 2],label="float32")
ax.plot(np.log10(np.abs(dx)), log_errors,label="float64")
ax.plot(np.log10(np.abs(mdx)), mlog_errors,dashes=[3, 2],label="float128")
ax.set_xlim(-0.1, -max_val)  # decreasing time
ax.set_xlabel('log(step size)')


ax.set_ylabel('log(error)')
ax.set_title('Error in second order finite difference approximation of the derivative of $x^3$')
ax.legend()
fig.savefig("/home/himanshu/Desktop/master_project/thesis/images/x^3_error_order2.png",bbox_inches='tight')


# ## Second derivative

# In[175]:



max_val = 6
dx = 0.1**np.arange(0.1, max_val,0.1)
error = np.zeros(len(dx))
error = -((1+dx)**4-2+(1-dx)**4)/(dx**2) + 12.0
log_errors = np.log10(np.abs(error))


mdx = 0.1**np.arange(0.1, max_val,0.1,dtype=np.longdouble)
merror = np.zeros(len(mdx),dtype=np.longdouble)
merror = -((1+mdx)**4-2+(1-mdx)**4)/(mdx**2) + 12.0
mlog_errors = np.log10(np.abs(merror))


ldx = 0.1**np.arange(0.1, max_val,0.1,dtype='f')
lerror = np.zeros(len(ldx),dtype='f')
lerror = -((1+ldx)**4-2+(1-ldx)**4)/(ldx**2) + 12.0
llog_errors = np.log10(np.abs(lerror))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(np.abs(ldx)), llog_errors,dashes=[1, 2],label="float32")
ax.plot(np.log10(np.abs(dx)), log_errors,label="float64")
ax.plot(np.log10(np.abs(mdx)), mlog_errors,dashes=[3, 2],label="float128")
ax.set_xlim(-0.1, -max_val)  # decreasing time
ax.set_xlabel('log(step size)')


ax.set_ylabel('log(error)')
ax.set_title('Error in second order finite difference approximation of the derivative of $x^3$')
ax.legend()
# fig.savefig("/home/himanshu/Desktop/master_project/thesis/images/x^3_errord2_order1.png",bbox_inches='tight')


# # $\frac{1}{x}$ error plot

# ## First order at x=1

# In[176]:


max_val = 12

def func(x):
    return 1/x
def funcd(x):
    return -1/x**2


derivative_at = np.float128(1)

dx = 0.1**np.arange(0.1, max_val,0.1)
error = np.zeros(len(dx))
error = -(func(derivative_at+dx)-func(derivative_at))/dx + funcd(derivative_at)
log_errors = np.log10(np.abs(error))

mdx = 0.1**np.arange(0.1, max_val,0.1,dtype=np.longdouble)
merror = np.zeros(len(mdx),dtype=np.longdouble)
merror = -(func(derivative_at+mdx)-func(derivative_at))/mdx + funcd(derivative_at)
mlog_errors = np.log10(np.abs(merror))

ldx = 0.1**np.arange(0.1, max_val,0.1,dtype='f')
lerror = np.zeros(len(ldx),dtype='f')
lerror = -(func(derivative_at+ldx)-func(derivative_at))/ldx + funcd(derivative_at)
llog_errors = np.log10(np.abs(lerror))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(np.abs(ldx)), llog_errors,dashes=[1, 2],label="float32")
ax.plot(np.log10(np.abs(dx)), log_errors,label="float64")
ax.plot(np.log10(np.abs(mdx)), mlog_errors,dashes=[3, 2],label="float128")
ax.set_xlim(-0.1, -max_val)  # decreasing time
ax.set_xlabel('log(step size)')

ax.set_ylabel('log(error)')
ax.set_title(f'Error in first order finite difference approximation of the derivative of $1/x$ at x={derivative_at}')

ax.legend()
# ax.grid(True)
fig.savefig("/home/himanshu/Desktop/master_project/thesis/images/1_x_error_at_1.png",bbox_inches='tight')


# # at x=0.1

# In[180]:


max_val = 12

def func(x):
    return 1/x
def funcd(x):
    return -1/x**2


derivative_at = np.float128(.1)


dx = np.float128(.1)**np.arange(0.1, max_val,0.1)
error = np.zeros(len(dx))
error = -(func(derivative_at+dx)-func(derivative_at))/dx + funcd(derivative_at)
log_errors = np.log10(np.abs(error))

mdx = np.float128(.1)**np.arange(0.1, max_val,0.1,dtype=np.longdouble)
merror = np.zeros(len(mdx),dtype=np.longdouble)
merror = -(func(derivative_at+mdx)-func(derivative_at))/mdx + funcd(derivative_at)
mlog_errors = np.log10(np.abs(merror))

ldx = np.float128(.1)**np.arange(0.1, max_val,0.1,dtype='f')
lerror = np.zeros(len(ldx),dtype='f')
lerror = -(func(derivative_at+ldx)-func(derivative_at))/ldx + funcd(derivative_at)
llog_errors = np.log10(np.abs(lerror))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(np.abs(ldx)), llog_errors,dashes=[1, 2],label="float32")
ax.plot(np.log10(np.abs(dx)), log_errors,label="float64")
ax.plot(np.log10(np.abs(mdx)), mlog_errors,dashes=[3, 2],label="float128")
ax.set_xlim(-0.1, -max_val)  # decreasing time
ax.set_xlabel('log(step size)')

ax.set_ylabel('log(error)')
ax.set_title(f'Error in first order finite difference approximation of the derivative of $1/x$ at x={derivative_at}')

ax.legend()
# ax.grid(True)
fig.savefig("/home/himanshu/Desktop/master_project/thesis/images/1_x_error_at_p1.png",bbox_inches='tight')


# ## at x = 0.01

# In[178]:


max_val = 12

def func(x):
    return 1/x
def funcd(x):
    return -1/x**2


derivative_at = np.float128(.01)


dx = 0.1**np.arange(0.1, max_val,0.1)
error = np.zeros(len(dx))
error = -(func(derivative_at+dx)-func(derivative_at))/dx + funcd(derivative_at)
log_errors = np.log10(np.abs(error))

mdx = 0.1**np.arange(0.1, max_val,0.1,dtype=np.longdouble)
merror = np.zeros(len(mdx),dtype=np.longdouble)
merror = -(func(derivative_at+mdx)-func(derivative_at))/mdx + funcd(derivative_at)
mlog_errors = np.log10(np.abs(merror))

ldx = 0.1**np.arange(0.1, max_val,0.1,dtype='f')
lerror = np.zeros(len(ldx),dtype='f')
lerror = -(func(derivative_at+ldx)-func(derivative_at))/ldx + funcd(derivative_at)
llog_errors = np.log10(np.abs(lerror))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(np.abs(ldx)), llog_errors,dashes=[1, 2],label="float32")
ax.plot(np.log10(np.abs(dx)), log_errors,label="float64")
ax.plot(np.log10(np.abs(mdx)), mlog_errors,dashes=[3, 2],label="float128")
ax.set_xlim(-0.1, -max_val)  # decreasing time
ax.set_xlabel('log(step size)')

ax.set_ylabel('log(error)')
ax.set_title(f'Error in first order finite difference approximation of the derivative of $1/x$ at x={derivative_at}')

ax.legend()
# ax.grid(True)
fig.savefig("/home/himanshu/Desktop/master_project/thesis/images/1_x_error_at_p01.png",bbox_inches='tight')


# In[179]:


mdx


# # Floating point errors

# In[227]:


a32 = np.float32(1/3)
a64 = np.float64(1/3)
a128 = np.float128(1.0)/3


# In[250]:



def add32(x,n):
    for i in range(n):
        x = x + np.float32(1.0)   
    return x

def add64(x,n):
    for i in range(n):
        x = x + 1.0
    return x
        
def add128(x,n):
    for i in range(n):
        x = x + np.float128(1.0)   
    return x



def mul32(x,n):
    for i in range(n):
        x = x * np.float128(3.0)   
        x = x * np.float32(1/3)   
    return x

def mul64(x,n):
    for i in range(n):
        x = x * np.float128(3.0)
        x = x * np.float64(1/3)
    return x
        
def mul128(x,n):
    for i in range(n):
        x = x * np.float128(3.0)   
        x = x * np.float128(1)/3   
    return x


# In[231]:


n = 10000

print("additions")
print(add32(a32,n),"---",a32 + 1*n)
print(add64(a64,n),"---",a64 + 1*n)
print(add128(a128,n),"---",a128 + np.float128(1)*n)


# In[259]:


n = 10000

print("multiplication")
print(mul32(a32,n))
print(a32)
print(mul64(a64,n))
print(a64)
print(mul128(a128,n))
print(a128)


# ## Number of floats in 8 GB

# In[194]:


print('{:.5E}'.format(1e9*8/64))


# In[196]:


np.sqrt(1e9)


# In[ ]:




