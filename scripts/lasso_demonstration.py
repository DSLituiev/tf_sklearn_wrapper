import numpy as numpy
import matplotlib.pyplot as plt
%matplotlib inline

x_ = np.arange(-10,10,0.2)
y_ = np.arange(-10,10,0.2)
x,y = np.meshgrid(x_,y_)

p = np.exp( - (x**2 + y**2))
plt.contour(x,y,p)
# p.shape

%matplotlib inline
x_ = np.arange(-10,10,0.2)
y_ = np.arange(-10,10,0.2)
x,y = np.meshgrid(x_,y_)

p = np.exp( - (abs(x) + abs(y)))
plt.contour(x,y,p)
# p.shape

