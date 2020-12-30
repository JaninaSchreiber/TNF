import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define the function that we are interested in
def sixhump(x):
    return ((4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1]
            + (-4 + 4*x[1]**2) * x[1] **2)

# Make a grid to evaluate the function (for plotting)
x = np.linspace(-2, 2)
y = np.linspace(-1, 1)
xg, yg = np.meshgrid(x, y)

# create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xg, yg, sixhump([xg, yg]), rstride=1, cstride=1,
                       cmap=plt.cm.jet, linewidth=0, antialiased=False)
ax.autoscale()

# set label and title
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$f(x_1, x_2)$')
#ax.set_title('Six-hump Camelback function')

#save figure
plt.savefig("sixhumpdomain.png")


x = np.arange(-2,0, 0.04)
y = np.arange(-1,-0.5, 0.01)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y,sixhump([x, y]), c=sixhump([x, y]), vmin=-1.03, vmax=5, cmap=jet)
ax.autoscale()

ax.set_xlabel('\n'+r'$x_1$', linespacing=3)
ax.set_ylabel('\n'+r'$x_2$', linespacing=3)
ax.set_zlabel('\n'+r'$f(x_1, x_2)$', linespacing=3)
#ax.set_title('Six-hump Camelback func

plt.savefig('/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/EMEWS-scripts/testcase/pictures/domain/rs_domain.png')
