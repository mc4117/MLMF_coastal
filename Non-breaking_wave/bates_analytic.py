"""
 This file calculates the analytical result for the non-breaking wave test case using the result published in

 Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010.  A simple inertial formulation of the shallow water
 equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33–45.
 
 The uncertain parameter here is the Manning friction coefficient and 
 the variable of interest is the water elevation height at four different locations.
"""

import numpy as np
import math
import pylab as plt
from scipy import interpolate

t = 3600 #to look at solution after 3600s
C = 0
u = 1

def h(x, n):
    # analytical solution
    tau = x - (u*t)
    return ((-7/3)*(n**2)*(u**2)*tau)**(3/7)


dx = 5
x = np.linspace(0, 5000, np.int((5000/dx)+1))

exp_list_1 = []
exp_list_2 = []
exp_list_3 = []
exp_list_4 = []
exp_list_5 = []

n = -1

n_list = []

for _ in range(10000):

    while n < 0:
        # ensure manning coefficient not less than 0
        n = np.random.normal(0.03, 0.01)

    if n < 0:
        stop

    n_list.append(n)
    new_h = [h(i, n) for i in x]
    
    h_interp = interpolate.interp1d(x, new_h)

    # extract outputs of interest
    exp_list_1.append(h_interp(1000))
    exp_list_2.append(h_interp(1500))
    exp_list_3.append(h_interp(2000))
    exp_list_4.append(h_interp(2500))

    n = -1

exp_list_float_1 = [float(i) for i in exp_list_1]
exp_list_float_2 = [float(i) for i in exp_list_2]
exp_list_float_3 = [float(i) for i in exp_list_3]
exp_list_float_4 = [float(i) for i in exp_list_4]

# plot histograms
plt.hist(n_list)
plt.show()
plt.hist(exp_list_float_1)
plt.show()
plt.hist(exp_list_float_2)
plt.show()
plt.hist(exp_list_float_3)
plt.show()
plt.hist(exp_list_float_4)
plt.show()

# calculate estimators for analytical result
print(np.mean(exp_list_1))
print(np.mean(exp_list_2))
print(np.mean(exp_list_3))
print(np.mean(exp_list_4))

# analytic [2.0479628449800495, 1.8688323192893563, 1.6632454544788324, 1.4164983561938571]
