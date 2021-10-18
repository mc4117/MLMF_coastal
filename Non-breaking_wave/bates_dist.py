"""
Calculate MLMF and MC cumulative distribution function (CDF) for the non-breaking wave test case 
first presented in

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010.  A simple inertial formulation of the shallow water
equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33–45.

Note no output files have been included in this repository so all csv files must be produced using 
the files in the Non-breaking_wave folder
"""

import pandas as pd
import pylab as plt
import numpy as np

from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator

from scipy.interpolate import interp1d

import matplotlib
font = {'size'   : 13}
matplotlib.rc('font', **font)


colorlist = ['red', 'green', 'blue', 'orange']

# gamma values to increase correlation for the second model run of this test case
gamma_2 = [[1, 1, 1, 1], [0.5376498514042958, 0.47330671791540524, 0.4168681540301937, -2.11835767077297], [0.7474996491222335, 1.1150068363667236, 0.8042951886454661, 0.3573408572908365], [0.9134562722583787, 0.8634835383336112, 1.2111119956328358, 1.136181531899857], [0.9652208761609411, 0.946850462765506, 0.9429444700775697, 1.257629331058035], [0.9823606401833977, 0.9773821762898511, 0.9817372905621558, 1.177033625603634], [0.9982206828721528, 0.9982271682675371, 0.9993702438299009, 1.0007586500618146]]

# modified correlation values for the second model run of this test case
rho_list = [[0.9900355613881667, 0.984043242649586, 0.9841512453662701, -0.18303632086782634], [0.9995471319908507, 0.9985160453591049, 0.9953498221362023, -0.957906075985963], [0.9970717889714953, -0.9901497556429639, -0.9962553028595251, -0.988693983704796], [0.9975972748665263, 0.9892876244970125, -0.9423853963651386, -0.9044001852854658], [0.9898371661951498, 0.9933217961681196, 0.9851494696410463, 0.6640139323508047], [0.9719258169474536, 0.9708120928186338, 0.97138001636409, 0.4857531490323038], [0.9284603253090108, 0.9072056021218708, 0.9409747548516895, 0.9769860885931438]]

f = open('../job_bates.o4477723').read().split('\n')

# read in output values
exec("values1 = " + f[154])
values1_arr = np.array(values1)

exec("values0 = " + f[152])
values0_arr = np.array(values0)

k_list = [0, 1, 2, 3]
# calculate alpha in control variate
alpha_list = []

for i in range(4):
    k = k_list[i]
    
    rho = [x[k] for x in rho_list]

    # calculate the variances
    variance0 = [np.var(varlist) for varlist in [x[k] for x in values0_arr]]
    variance1 = [np.var(varlist) for varlist in [x[k] for x in values1_arr]]

    # optimum value to combine low fidelity and high fidelity model
    alpha = [-rho[j]*np.sqrt(variance0[j]/variance1[j]) for j in range(len(rho))]
    alpha_list.append(alpha)
    
## Sample function for u
def unif():
    return np.random.uniform(0, 1.0, 1)[0]

## Read in monte carlo data
def cdf(loc_no):

    # function to estimate cdf at given location

    ## Read in low fidelity data
    output4_lf_orig = pd.read_csv('output_lf8.csv')[loc_no]
    output5_lf_orig = pd.read_csv('output_lf16.csv')[loc_no]
    output6_lf_orig = pd.read_csv('output_lf32.csv')[loc_no]
    output7_lf_orig = pd.read_csv('output_lf64.csv')[loc_no]
    output8_lf_orig = pd.read_csv('output_lf128.csv')[loc_no]
    output9_lf_orig = pd.read_csv('output_lf256.csv')[loc_no]
    output10_lf_orig = pd.read_csv('output_lf512.csv')[loc_no]
    output11_lf_orig = pd.read_csv('output_lf1024.csv')[loc_no]

    output4_2lf_orig = pd.read_csv('output_lf_28.csv')[loc_no]
    output5_2lf_orig = pd.read_csv('output_lf_216.csv')[loc_no]
    output6_2lf_orig = pd.read_csv('output_lf_232.csv')[loc_no]
    output7_2lf_orig = pd.read_csv('output_lf_264.csv')[loc_no]
    output8_2lf_orig = pd.read_csv('output_lf_2128.csv')[loc_no]
    output9_2lf_orig = pd.read_csv('output_lf_2256.csv')[loc_no]
    output10_2lf_orig = pd.read_csv('output_lf_2512.csv')[loc_no]
    output11_2lf_orig = pd.read_csv('output_lf_21024.csv')[loc_no]

    # read in optimum number of samples required
    f0_2 = open('bates_comb_1e3_2nd_2.txt').read().split('\n')
    f0 = open('bates_comb_1e3_2nd.txt').read().split('\n')
    
    k = np.int(float(loc_no))
    
    N_MLMF_HF_0_0 = [np.int(i) for i in f0[1+9*k][1:-1].split(',')]
    N_MLMF_HF_0 = [np.int(i) for i in f0_2[1+9*k][1:-1].split(',')]    

    max_list = [max(min(N_MLMF_HF_0_0, N_MLMF_HF_0)[i], 200) for i in range(len(N_MLMF_HF_0_0))]

    # estimate the expectation used in the control variate from the low fidelity samples
    output7_lf_exp2 = np.mean([gamma_2[2][k]*output7_2lf_orig[i] for i in range(len(output7_2lf_orig))])
    output8_lf_exp2 = np.mean([gamma_2[3][k]*output8_2lf_orig[i]-output7_2lf_orig[i] for i in range(min(len(output8_2lf_orig), len(output7_2lf_orig)))])
    output9_lf_exp2 = np.mean([gamma_2[4][k]*output9_2lf_orig[i]-output8_2lf_orig[i] for i in range(min(len(output9_2lf_orig), len(output8_2lf_orig)))])
    output10_lf_exp2 = np.mean([gamma_2[5][k]*output10_2lf_orig[i]-output9_2lf_orig[i] for i in range(min(len(output10_2lf_orig), len(output9_2lf_orig)))])
    output11_lf_exp2 = np.mean([gamma_2[6][k]*output11_2lf_orig[i]-output10_2lf_orig[i] for i in range(min(len(output11_2lf_orig), len(output10_2lf_orig)))])

    outputlist2_exp_nan = [output7_lf_exp2, output8_lf_exp2, output9_lf_exp2, output10_lf_exp2, output11_lf_exp2]

    # extract NHF low fidelity samples from each model run and sort them
    output5_lf = np.sort(output5_lf_orig[-max_list[0]-max_list[1]:])
    output6_lf = np.sort(output6_lf_orig[-max_list[1]-max_list[2]:])
    output7_lf = np.sort(output7_lf_orig[-max_list[2]-max_list[3]:])
    output8_lf = np.sort(output8_lf_orig[-max_list[3]-max_list[4]:])
    output9_lf = np.sort(output9_lf_orig[-max_list[4]-max_list[5]:])
    output10_lf = np.sort(output10_lf_orig[-max_list[5]-max_list[6]:])
    output11_lf = np.sort(output11_lf_orig[-max_list[6]:])

    output5_2lf = np.sort(output5_2lf_orig[-max_list[0]-max_list[1]:])
    output6_2lf = np.sort(output6_2lf_orig[-max_list[1]-max_list[2]:])
    output7_2lf = np.sort(output7_2lf_orig[-max_list[2]-max_list[3]:])
    output8_2lf = np.sort(output8_2lf_orig[-max_list[3]-max_list[4]:])
    output9_2lf = np.sort(output9_2lf_orig[-max_list[4]-max_list[5]:])
    output10_2lf = np.sort(output10_2lf_orig[-max_list[5]-max_list[6]:])
    output11_2lf = np.sort(output11_2lf_orig[-max_list[6]:])

    ## Read in high fidelity data and extract NHF high fidelity samples from each model run and sort them
    output5_hf = np.sort(pd.read_csv('output_hf16.csv')[loc_no][-max_list[0]-max_list[1]:])
    output6_hf = np.sort(pd.read_csv('output_hf32.csv')[loc_no][-max_list[1]-max_list[2]:])
    output7_hf = np.sort(pd.read_csv('output_hf64.csv')[loc_no][-max_list[2]-max_list[3]:])
    output8_hf = np.sort(pd.read_csv('output_hf128.csv')[loc_no][-max_list[3]-max_list[4]:])
    output9_hf = np.sort(pd.read_csv('output_hf256.csv')[loc_no][-max_list[4]-max_list[5]:])
    output10_hf = np.sort(pd.read_csv('output_hf512.csv')[loc_no][-max_list[5]-max_list[6]:])
    output11_hf = np.sort(pd.read_csv('output_hf1024.csv')[loc_no][-max_list[6]:])

    output5_2hf = np.sort(pd.read_csv('output_hf_216.csv')[loc_no][-max_list[0]-max_list[1]:])
    output6_2hf = np.sort(pd.read_csv('output_hf_232.csv')[loc_no][-max_list[1]-max_list[2]:])
    output7_2hf = np.sort(pd.read_csv('output_hf_264.csv')[loc_no][-max_list[2]-max_list[3]:])
    output8_2hf = np.sort(pd.read_csv('output_hf_2128.csv')[loc_no][-max_list[3]-max_list[4]:])
    output9_2hf = np.sort(pd.read_csv('output_hf_2256.csv')[loc_no][-max_list[4]-max_list[5]:])
    output10_2hf = np.sort(pd.read_csv('output_hf_2512.csv')[loc_no][-max_list[5]-max_list[6]:])
    output11_2hf = np.sort(pd.read_csv('output_hf_21024.csv')[loc_no][-max_list[6]:])

    # create inverse samples
    x_ilist_mlmf = []

    for i in range(200000):
        x_i = 0
        u = unif()
    
        x_i += output7_2hf[np.int(u*len(output7_2hf))] + alpha_list[k][2]*(gamma_2[2][k]*
                    (output7_2lf[np.int(u*len(output7_2lf))]) - outputlist2_exp_nan[0]) 

        x_i += output8_2hf[np.int(u*len(output8_2hf))] - output7_hf[np.int(u*len(output7_hf))] + \
            alpha_list[k][3]*((gamma_2[3][k]*(output8_2lf[np.int(u*len(output8_2lf))]) - 
                       output7_lf[np.int(u*len(output7_lf))])-outputlist2_exp_nan[1])
        
        x_i += output9_2hf[np.int(u*len(output9_2hf))] - output8_hf[np.int(u*len(output8_hf))] + \
            alpha_list[k][4]*((gamma_2[4][k]*(output9_2lf[np.int(u*len(output9_2lf))]) - 
                       output8_lf[np.int(u*len(output8_lf))])-outputlist2_exp_nan[2])
        
        x_i += output10_2hf[np.int(u*len(output10_2hf))] - output9_hf[np.int(u*len(output9_hf))] + \
            alpha_list[k][5]*((gamma_2[5][k]*(output10_2lf[np.int(u*len(output10_2lf))]) - 
                       output9_lf[np.int(u*len(output9_lf))])-outputlist2_exp_nan[3])

        x_i += output11_2hf[np.int(u*len(output11_2hf))] - output10_hf[np.int(u*len(output10_hf))] + \
            alpha_list[k][6]*((gamma_2[6][k]*(output11_2lf[np.int(u*len(output11_2lf))]) - 
                       output10_lf[np.int(u*len(output10_lf))])-outputlist2_exp_nan[4])        

        x_ilist_mlmf.append(x_i)

    
    return x_ilist_mlmf

# read in Monte Carlo outputs
output_0 = pd.read_csv('../bates_test_mc/output_0.csv')['0']
output_1 = pd.read_csv('../bates_test_mc/output_1.csv')['0']
output_2 = pd.read_csv('../bates_test_mc/output_2.csv')['0']
output_3 = pd.read_csv('../bates_test_mc/output_3.csv')['0']

# apply inverse transform sampling method at each location
x_ilist_mlmf_0 = cdf('0')
x_ilist_mlmf_1 = cdf('1')
x_ilist_mlmf_2 = cdf('2')
x_ilist_mlmf_3 = cdf('3')

# plot pdfs
plt.hist(output_0, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(x_ilist_mlmf_0, bins = 101, histtype=u'step', density = True, label = 'MLMF')
plt.legend()
plt.show()

plt.hist(output_1, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(x_ilist_mlmf_1, bins = 101, histtype=u'step', density = True, label = 'MLMF')
plt.legend()
plt.show()

plt.hist(output_2, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(x_ilist_mlmf_2, bins = 101, histtype=u'step', density = True, label = 'MLMF')
plt.legend()
plt.show()

plt.hist(output_3, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(x_ilist_mlmf_3, bins = 101, histtype=u'step', density = True, label = 'MLMF')
plt.legend()
plt.show()

hist_out_0, bins_out_0 = np.histogram(np.asarray(output_0), bins = 101, density = True)
hist_mlmf_0, bins_mlmf_0 = np.histogram(np.asarray(x_ilist_mlmf_0), bins = 101, density = True)
hist_out_1, bins_out_1 = np.histogram(np.asarray(output_1), bins = 101, density = True)
hist_mlmf_1, bins_mlmf_1 = np.histogram(np.asarray(x_ilist_mlmf_1), bins = 101, density = True)
hist_out_2, bins_out_2 = np.histogram(np.asarray(output_2), bins = 101, density = True)
hist_mlmf_2, bins_mlmf_2 = np.histogram(np.asarray(x_ilist_mlmf_2), bins = 101, density = True)
hist_out_3, bins_out_3 = np.histogram(np.asarray(output_3), bins = 101, density = True)
hist_mlmf_3, bins_mlmf_3 = np.histogram(np.asarray(x_ilist_mlmf_3), bins = 101, density = True)

## calculate cumulative distributions
cum_hist_out_0 = np.cumsum(hist_out_0)
cum_hist_mlmf_0 = np.cumsum(hist_mlmf_0)
cum_hist_out_1 = np.cumsum(hist_out_1)
cum_hist_mlmf_1 = np.cumsum(hist_mlmf_1)
cum_hist_out_2 = np.cumsum(hist_out_2)
cum_hist_mlmf_2 = np.cumsum(hist_mlmf_2)
cum_hist_out_3 = np.cumsum(hist_out_3)
cum_hist_mlmf_3 = np.cumsum(hist_mlmf_3)

# plot cdfs
fig, ax = plt.subplots()
ax.plot(bins_mlmf_0[1:], cum_hist_mlmf_0/cum_hist_mlmf_0[-1],  color = colorlist[0])
ax.plot(bins_out_0[1:], cum_hist_out_0/cum_hist_out_0[-1], '--', color = colorlist[0])
ax.plot(bins_mlmf_1[1:], cum_hist_mlmf_1/cum_hist_mlmf_1[-1],  color = colorlist[1])
ax.plot(bins_out_1[1:], cum_hist_out_1/cum_hist_out_1[-1], '--', color = colorlist[1])
ax.plot(bins_mlmf_2[1:], cum_hist_mlmf_2/cum_hist_mlmf_2[-1],  color = colorlist[2])
ax.plot(bins_out_2[1:], cum_hist_out_2/cum_hist_out_2[-1], '--', color = colorlist[2])
ax.plot(bins_mlmf_3[1:], cum_hist_mlmf_3/cum_hist_mlmf_3[-1],  color = colorlist[3])
ax.plot(bins_out_3[1:], cum_hist_out_3/cum_hist_out_3[-1], '--', color = colorlist[3])
          
ax.plot(bins_mlmf_0[1], cum_hist_mlmf_0[1]/cum_hist_mlmf_0[-1], 'k')
ax.plot(bins_out_0[1], cum_hist_out_0[1]/cum_hist_out_0[-1], 'k--')

lines = ax.get_lines()
legend1 = pyplot.legend([lines[i] for i in [0, 2, 4, 6]], ['x=1000m', 'x=1500m', 'x=2000m', 'x=2500m'], loc =2 )
legend2 = pyplot.legend([lines[i] for i in [8, 9]], ['MLMF', 'MC'], loc = 4)
ax.add_artist(legend1)
ax.add_artist(legend2)

ax.set_xlabel('Output')
ax.set_ylabel(r'$\mathbb{P}(X \leq x)$')
plt.show()


# calculate errors between Monte Carlo and MLMF cdfs at all locations
a_mlmf = cum_hist_mlmf_0/cum_hist_mlmf_0[-1]

b = cum_hist_out_0/cum_hist_out_0[-1]

error = sum([(a_mlmf[i] - b[i])**2 for i in range(len(a_mlmf))])/(sum([a_mlmf[i]**2 for i in range(len(a_mlmf))]))

print('1')

print(error)

a_mlmf = cum_hist_mlmf_1/cum_hist_mlmf_1[-1]

b = cum_hist_out_1/cum_hist_out_1[-1]

error = sum([(a_mlmf[i] - b[i])**2 for i in range(len(a_mlmf))])/(sum([a_mlmf[i]**2 for i in range(len(a_mlmf))]))

print('2')

print(error)

a_mlmf = cum_hist_mlmf_2/cum_hist_mlmf_2[-1]

b = cum_hist_out_2/cum_hist_out_2[-1]

error = sum([(a_mlmf[i] - b[i])**2 for i in range(len(a_mlmf))])/(sum([a_mlmf[i]**2 for i in range(len(a_mlmf))]))

print('3')

print(error)

a_mlmf = cum_hist_mlmf_3/cum_hist_mlmf_3[-1]

b = cum_hist_out_3/cum_hist_out_3[-1]

error = sum([(a_mlmf[i] - b[i])**2 for i in range(len(a_mlmf))])/(sum([a_mlmf[i]**2 for i in range(len(a_mlmf))]))

print('4')

print(error)