"""
Calculate MLMF and MC cumulative distribution function (CDF) for the Carrier-Greenspan test case 
first presented in

Carrier, G.F., Greenspan, H.P., 1958. Water waves of finite amplitude on a sloping beach. 
Journal of Fluid Mechanics 4, 97–109.

Note no output files have been included in this repository so all csv files must be produced using 
the files in the Carrier-Greenspan folder
"""

import pandas as pd
import pylab as plt
import numpy as np

from scipy.interpolate import interp1d

import matplotlib
font = {'size'   : 13}
matplotlib.rc('font', **font)

## Read in monte carlo data
output = pd.read_csv('../carrier_runup_mc/total_output.csv')['0']

# gamma values to increase correlation for the second model run of this test case
gamma_2 = [[1], [0.4343779257506352], [0.894765703377151], [0.945357568664327], [0.9430495618872295]]

## Read in first sample low fidelity data and sort it 
output5_lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf32.csv')['0']]
output6_lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf64.csv')['0']]
output7_lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf128.csv')['0']]
output8_lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf256.csv')['0']]
output9_lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf512.csv')['0']]

## Read in second sample low fidelity data and sort it
output5_2lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf_232.csv')['0']]
output6_2lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf_264.csv')['0']]
output7_2lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf_2128.csv')['0']]
output8_2lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf_2256.csv')['0']]
output9_2lf_orig = [float(i[1:-1]) for i in pd.read_csv('../carrier_runup_mod2/output_lf_2512.csv')['0']]

# modified correlation values for the second model run of this test case
rho_list = [[0.9665660828123629], [-0.677708496275484], [-0.2637138953385968], [0.575437679065745], [0.7328089723571111]]

f = open('../job_carrier_runup_mod.o4477688').read().split('\n')

# read in output values
exec("values1 = " + f[57])
values1_arr = np.array(values1)

exec("values0 = " + f[55])
values0_arr = np.array(values0)

# calculate alpha in control variate
alpha_list = []

k = 0
    
rho = [x[k] for x in rho_list]

# calculate the variances
variance0 = [np.var(varlist) for varlist in [x[k] for x in values0_arr]]   
variance1 = [np.var(varlist) for varlist in [x[k] for x in values1_arr]]
# optimum value to combine low fidelity and high fidelity model
alpha_list = [-rho[j]*np.sqrt(variance0[j]/variance1[j]) for j in range(len(rho))]

# estimate the expectation used in the control variate from the low fidelity samples
output6_lf_exp2 = np.mean([gamma_2[1][0]*output6_2lf_orig[i] for i in range(len(output6_2lf_orig))])
output7_lf_exp2 = np.mean([gamma_2[2][0]*output7_2lf_orig[i]-output6_2lf_orig[i] for i in range(min(len(output7_2lf_orig), len(output6_2lf_orig)))])
output8_lf_exp2 = np.mean([gamma_2[3][0]*output8_2lf_orig[i]-output7_2lf_orig[i] for i in range(min(len(output8_2lf_orig), len(output7_2lf_orig)))])
output9_lf_exp2 = np.mean([gamma_2[4][0]*output9_2lf_orig[i]-output8_2lf_orig[i] for i in range(min(len(output9_2lf_orig), len(output8_2lf_orig)))])


# check for any nan values
outputlist2 = [output6_lf_exp2, output7_lf_exp2, output8_lf_exp2, output9_lf_exp2]
nancheck2 = np.isnan(outputlist2)

outputlist2_exp_nan = []

for i in range(len(nancheck2)):
    if nancheck2[i]:
        outputlist2_exp_nan.append(0)
    elif nancheck2[i] == False:
        outputlist2_exp_nan.append(outputlist2[i])

# read in optimum number of samples required
f0 = open('../carrier_runup_mod2/carrier_fix_comb_1e3.txt').read().split('\n')

N_MLMF_LF = [np.int(i) for i in f0[3][1:-1].split(',')]
N_MLMF_HF = [np.int(i) for i in f0[1][1:-1].split(',')]

N_MLMF_LF_2 = [np.int(i) for i in f0[12][1:-1].split(',')]
N_MLMF_HF_2 = [np.int(i) for i in f0[10][1:-1].split(',')]

max_list = [max(min(N_MLMF_HF, N_MLMF_HF_2)[i], 480) for i in range(len(N_MLMF_HF_2))]

# extract NHF low fidelity samples from each model run and sort them. Ensure these samples are not the same 
# as those already used to estimate the low fidelity expectation in the control variate
output5_lf = np.sort(output5_lf_orig[-max_list[0]-max_list[1]:])
output6_lf = np.sort(output6_lf_orig[-max_list[1]-max_list[2]:])
output7_lf = np.sort(output7_lf_orig[-max_list[2]-max_list[3]:])
output8_lf = np.sort(output8_lf_orig[-max_list[3]-max_list[4]:])
output9_lf = np.sort(output9_lf_orig[-max_list[4]:])


output5_2lf = np.sort(output5_2lf_orig[-max_list[0]-max_list[1]:])
output6_2lf = np.sort(output6_2lf_orig[-max_list[1]-max_list[2]:])
output7_2lf = np.sort(output7_2lf_orig[-max_list[2]-max_list[3]:])
output8_2lf = np.sort(output8_2lf_orig[-max_list[3]-max_list[4]:])
output9_2lf = np.sort(output9_2lf_orig[-max_list[4]:])


## Read in high fidelity data from each model run and sort it 
output5_hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf32.csv')['0'][-max_list[0]-max_list[1]:])
output6_hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf64.csv')['0'][-max_list[1]-max_list[2]:])
output7_hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf128.csv')['0'][-max_list[2]-max_list[3]:])
output8_hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf256.csv')['0'][-max_list[3]-max_list[4]:])
output9_hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf512.csv')['0'][-max_list[4]:])

output5_2hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf_232.csv')['0'][-max_list[0]-max_list[1]:])
output6_2hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf_264.csv')['0'][-max_list[1]-max_list[2]:])
output7_2hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf_2128.csv')['0'][-max_list[2]-max_list[3]:])
output8_2hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf_2256.csv')['0'][-max_list[3]-max_list[4]:])
output9_2hf = np.sort(pd.read_csv('../carrier_runup_mod2/output_hf_2512.csv')['0'][-max_list[4]:])

## Sample function for u
def unif():
    return np.random.uniform(0, 1.0, 1)[0]


# create inverse samples
x_ilist_mlmf = []

for i in range(200000):
    x_i = 0
    u = unif()
    
    x_i += output6_2hf[np.int(u*len(output6_2hf))] + alpha_list[1]*(gamma_2[1][0]*(output6_2lf[np.int(u*len(output6_2lf))]) - outputlist2_exp_nan[0])

    x_i += output7_2hf[np.int(u*len(output7_2hf))] - output6_hf[np.int(u*len(output6_hf))] + \
        alpha_list[2]*((gamma_2[2][0]*(output7_2lf[np.int(u*len(output7_2lf))]) - 
                       output6_lf[np.int(u*len(output6_lf))])-outputlist2_exp_nan[1])
        
    x_i += output8_2hf[np.int(u*len(output8_2hf))] - output7_hf[np.int(u*len(output7_hf))] + \
        alpha_list[3]*((gamma_2[3][0]*(output8_2lf[np.int(u*len(output8_2lf))]) - 
                       output7_lf[np.int(u*len(output7_lf))])-outputlist2_exp_nan[2])
        
    x_i += output9_2hf[np.int(u*len(output9_2hf))] - output8_hf[np.int(u*len(output8_hf))] + \
        alpha_list[4]*((gamma_2[4][0]*(output9_2lf[np.int(u*len(output9_2lf))]) - 
                       output8_lf[np.int(u*len(output8_lf))])-outputlist2_exp_nan[3])        

    x_ilist_mlmf.append(x_i)

# plot pdfs
plt.hist(output, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(x_ilist_mlmf, bins = 101, histtype=u'step', density = True, label = 'MLMF')
plt.legend()
plt.show()

hist_out, bins_out = np.histogram(np.asarray(output), bins = 101, density = True)
hist_mlmf, bins_mlmf = np.histogram(np.asarray(x_ilist_mlmf), bins = 101, density = True)

## calculate cumulative distributions
cum_hist_out = np.cumsum(hist_out)
cum_hist_mlmf = np.cumsum(hist_mlmf)

# plot cumulative distributions
plt.plot(bins_out[1:], cum_hist_out/cum_hist_out[-1], '--', color = 'blue', label = 'MC') 
plt.plot(bins_mlmf[1:], cum_hist_mlmf/cum_hist_mlmf[-1], color = 'blue', label = 'MLMF')
plt.plot([5.02, 5.483], [0.95, 0.95], 'b:')
plt.plot([5.483, 5.483], [-0.05, 0.95], 'k:', label = '95% prob')
plt.xlim([5.02, 5.55])
plt.ylim([-0.05, 1.05])
plt.legend(loc = 6)
plt.xlabel('Output')
plt.ylabel(r'$\mathbb{P}(X \leq x)$')
plt.show() 

# calculate error between Monte Carlo and MLMF CDFs
           
a_mlmf = cum_hist_mlmf/cum_hist_mlmf[-1]

b = cum_hist_out/cum_hist_out[-1]

error = sum([(a_mlmf[i] - b[i])**2 for i in range(len(a_mlmf))])/(sum([a_mlmf[i]**2 for i in range(len(a_mlmf))]))

print(error)
