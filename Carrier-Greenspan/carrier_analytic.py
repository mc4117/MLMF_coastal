"""
This file calculates the analytical result for the Carrier-Greenspan test case using the result published in

Carrier, G.F., Greenspan, H.P., 1958. Water waves of finite amplitude on a sloping beach. 
Journal of Fluid Mechanics 4, 97–109.
 
The uncertain parameter here is the slope and the variable of interest is the maximum run-up height.

"""

import numpy as np
import pylab as plt
from scipy.interpolate import interp1d

slope_list = []
eta_max = []

g = 9.81
h0 = 5
T = 32 # wave period

# set up bessel functions and non-dimensional solution
sigma = np.linspace(0, 20, 201)
lamda = np.linspace(0, 30, 301)

phi = np.zeros((len(sigma), len(lamda)))
dphi_dlamda = np.zeros((len(sigma), len(lamda)))
dphi_dsigma = np.zeros((len(sigma), len(lamda)))
v = np.zeros((len(sigma), len(lamda)))
x = np.zeros((len(sigma), len(lamda)))

for i in range(len(sigma)):
    for j in range(len(lamda)):
        s = sigma[i]
        l = lamda[j]
        
        # bessel function
        h = np.pi/100
        
        y = [np.exp(1j*s*np.sin(k)) for k in np.linspace(-np.pi, np.pi, 201)]
        J0 = (1/(2*np.pi))*(0.5*h)*(y[0]+ y[-1]+ 2*sum([y[l] for l in range(1, len(y) -1)]))
        
        y_1 = [np.exp(1j*(k-s*np.sin(k))) for k in np.linspace(-np.pi, np.pi, 201)]
        J0_prime = (1/(2*np.pi))*(0.5*h)*(y_1[0]+ y_1[-1]+ 2*sum([y_1[m] for m in range(1, len(y_1) -1)]))
        
        phi[i, j] = J0*np.cos(l)
        dphi_dlamda[i,j] = -J0*np.sin(l)
        dphi_dsigma[i,j] = -J0_prime*np.cos(l)


L, S = np.meshgrid(lamda, sigma)


# calculate analytical solution for different uncertain slope values
for _ in range(400000):
    
    slope = -1
    
    while slope < 0.005:
        slope = np.random.normal(2/50, 0.02)
        if slope <0.005:
            print(slope)
        else:
            slope_list.append(slope)

    L0 = (T**2)*g*slope
    
    A = 0.5*(slope**2.5)/(np.sqrt(128)*np.pi**3)*(T**2.5)*(g**1.25)*(h0**(-0.25))
    
    v = 1/S*A*dphi_dsigma; v[0,:] = 0
    eta = (A*dphi_dlamda/4-v**2/2).real

    eta_max.append(h0+slope*L0*eta.max(axis = 1)[0])

# plot histograms
plt.hist(eta_max)
plt.show()

# calculate estimators for analytical result
print(np.mean(eta_max))

# 5.437640571226524