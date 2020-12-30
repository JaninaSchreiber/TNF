import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 1 dimensional
fig = plt.figure()
ax = fig.add_subplot(111)

def mypolynom(x):
    return x[0]**2 + x[1]**2

plt.plot(xg[0], mypolynom([xg,0])[0] )
ax.set_xlabel('x(1)')
ax.set_ylabel('f(x(1),0)')
ax.set_title('Six-hump Camelback function')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xg, yg, mypolynom([xg, yg]), rstride=1, cstride=1,
                       cmap=plt.cm.jet, linewidth=0, antialiased=False)
plt.show()


x = np.arange(-20,20,0.01) 

def boltzmann_density(x, t, intv):
    return np.exp((-(x-1)**2)/t)/intv

def boltzmann(x, t):
    return np.sqrt(1/(2*np.pi*t))**3*4*np.pi*x**2*np.exp(-(x**2)/(2*t))

a01 = boltzmann_density(x, 0.1, 0.560497)
a1 = boltzmann_density(x, 1, 1.6289)
a5 = boltzmann_density(x, 5, 2.5108)
a10 = boltzmann_density(x, 10, 2.7302)

fig = plt.figure()
ax = fig.add_subplot(111)

#plt.plot(x, a01, label="T=0.1")
plt.plot(x,a1, label="T=1")
plt.plot(x,a10, label="T=10")
plt.plot(x,a100, label="T=100")

plt.legend()


def boltzmann_density2(x, t, intv):
    # -2 bis 5
    return np.exp((-(x**2+2x)**2)/t)/intv


#a01 = boltzmann_density2(x, 0.1, 12345.77)
a1 = boltzmann_density2(x, 1, 4.81803)
a10 = boltzmann_density2(x, 10, 6.1945)
a100 = boltzmann_density2(x, 100, 17.8115)
a200 = boltzmann_density2(x, 200, 24.081)
