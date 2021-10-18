"""
Calculate MLMF cumulative distribution function (CDF) for the Myrtle Beach test case 

Note no output files have been included in this repository so all csv files must be produced using 
the files in the Myrtle_Beach folder.
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

## Sample function for u
def unif():
    return np.random.uniform(0, 1.0, 1)[0]

# read in key mlmf parameters
f = open('../myrtle_parameters.txt').read().split('\n')

exec("gamma_2 = " + f[3])
exec("alpha_list = " + f[1])
exec("N_HF = " + f[5])
exec("N_LF = " + f[7])

# read in high fidelity outputs
df_c_hf2 = pd.read_csv('../myrtle_compiled/df_c_hf2.csv')
df_c_hf4 = pd.read_csv('../myrtle_compiled/df_c_hf4.csv')
df_c_hf8 = pd.read_csv('../myrtle_compiled/df_c_hf8.csv')
df_c_hf16 = pd.read_csv('../myrtle_compiled/df_c_hf16.csv')

df_f_hf2 = pd.read_csv('../myrtle_compiled/df_f_hf2.csv')
df_f_hf4 = pd.read_csv('../myrtle_compiled/df_f_hf4.csv')
df_f_hf8 = pd.read_csv('../myrtle_compiled/df_f_hf8.csv')
df_f_hf16 = pd.read_csv('../myrtle_compiled/df_f_hf16.csv')

df_hf_2 = pd.concat([df_f_hf2, df_c_hf4]).reset_index(drop = True)
df_hf_4 = pd.concat([df_f_hf4, df_c_hf8]).reset_index(drop = True)
df_hf_8 = pd.concat([df_f_hf8, df_c_hf16]).reset_index(drop = True)
df_hf_16 = pd.concat([df_f_hf16]).reset_index(drop = True)

# read in low fidelity outputs
df_c_lf2 = pd.read_csv('../myrtle_compiled/df_c_lf2.csv')
df_c_lf4 = pd.read_csv('../myrtle_compiled/df_c_lf4.csv')
df_c_lf8 = pd.read_csv('../myrtle_compiled/df_c_lf8.csv')
df_c_lf16 = pd.read_csv('../myrtle_compiled/df_c_lf16.csv')

df_f_lf2 = pd.read_csv('../myrtle_compiled/df_f_lf2.csv')
df_f_lf4 = pd.read_csv('../myrtle_compiled/df_f_lf4.csv')
df_f_lf8 = pd.read_csv('../myrtle_compiled/df_f_lf8.csv')
df_f_lf16 = pd.read_csv('../myrtle_compiled/df_f_lf16.csv')

df_lf_2 = pd.concat([df_f_lf2, df_c_lf4]).reset_index(drop = True)
df_lf_4 = pd.concat([df_f_lf4, df_c_lf8]).reset_index(drop = True)
df_lf_8 = pd.concat([df_f_lf8, df_c_lf16]).reset_index(drop = True)
df_lf_16 = pd.concat([df_f_lf16]).reset_index(drop = True)

# read in extra low fidelity outputs for expectation in control variate
df_extra_f_lf2 = pd.read_csv('../myrtle_compiled/df_extra_f_lf_2.csv')
df_extra_f_lf4 = pd.read_csv('../myrtle_compiled/df_extra_f_lf_4.csv')
df_extra_f_lf8 = pd.read_csv('../myrtle_compiled/df_extra_f_lf_8.csv')
df_extra_f_lf16 = pd.read_csv('../myrtle_compiled/df_extra_f_lf_16.csv')

df_extra_c_lf2 = pd.read_csv('../myrtle_compiled/df_extra_c_lf_2.csv')
df_extra_c_lf4 = pd.read_csv('../myrtle_compiled/df_extra_c_lf_4.csv')
df_extra_c_lf8 = pd.read_csv('../myrtle_compiled/df_extra_c_lf_8.csv')
df_extra_c_lf16 = pd.read_csv('../myrtle_compiled/df_extra_c_lf_16.csv')

colorlist = ['red', 'blue', 'green', 'purple', 'orange', 'olive', 'brown', 'cyan']
labellist = ['3', '4', '6', '7', '5', '1', '8', '2']
# bed level
zb = [2.01286005, 0.57829945, 1.35842052, 1.70495462, 3.93406047, 2.44806309, 3.71737334, 2.57699322]


def cdf(loc_no):

    # function to estimate cdf at given location
    
    # extract optimum number of samples for low fidelity expectation estimator
    N_LF_loc = [N_LF[loc_no][i] + np.int(np.ceil(N_HF[loc_no][i])) for i in range(len(N_LF[loc_no]))]
    output2_f_lf_orig = df_extra_f_lf2[:N_LF_loc[0]][str(loc_no)]
    output4_f_lf_orig = df_extra_f_lf4[:N_LF_loc[1]][str(loc_no)]
    output8_f_lf_orig = df_extra_f_lf8[:N_LF_loc[2]][str(loc_no)]
    output16_f_lf_orig = df_extra_f_lf16[:N_LF_loc[3]][str(loc_no)]

    output2_c_lf_orig = df_extra_c_lf2[:N_LF_loc[0]][str(loc_no)]
    output4_c_lf_orig = df_extra_c_lf4[:N_LF_loc[1]][str(loc_no)]
    output8_c_lf_orig = df_extra_c_lf8[:N_LF_loc[2]][str(loc_no)]
    output16_c_lf_orig = df_extra_c_lf16[:N_LF_loc[3]][str(loc_no)]

    # estimate the expectation used in the control variate from the low fidelity samples
    output4_lf_exp2 = np.mean([gamma_2[1][loc_no]*output4_f_lf_orig[i]
                               for i in range(len(output4_c_lf_orig))])
    output8_lf_exp2 = np.mean([gamma_2[2][loc_no]*output8_f_lf_orig[i]-output8_c_lf_orig[i] 
                               for i in range(len(output8_c_lf_orig))])
    output16_lf_exp2 = np.mean([gamma_2[3][loc_no]*output16_f_lf_orig[i]-output16_c_lf_orig[i] 
                               for i in range(len(output16_c_lf_orig))])                              

    outputlist2_exp_nan = [output4_lf_exp2, output8_lf_exp2, output16_lf_exp2]
    
    # extract NHF low fidelity samples from each model run and sort them
    N_HF_loc = [np.int(np.ceil(N_HF[loc_no][i])) for i in range(len(N_HF[loc_no]))]

    output2_2lf = np.sort(df_lf_2[:N_HF_loc[0]][str(loc_no)])
    output4_2lf = np.sort(df_lf_4[:N_HF_loc[1]][str(loc_no)])
    output8_2lf = np.sort(df_lf_8[:N_HF_loc[2]][str(loc_no)])
    output16_2lf = np.sort(df_lf_16[:N_HF_loc[3]][str(loc_no)])
    
    output2_lf = np.sort(df_lf_2[N_HF_loc[0]: 2*N_HF_loc[0]][str(loc_no)])
    output4_lf = np.sort(df_lf_4[N_HF_loc[1]: 2*N_HF_loc[1]][str(loc_no)])
    output8_lf = np.sort(df_lf_8[N_HF_loc[2]: 2*N_HF_loc[2]][str(loc_no)])
    output16_lf = np.sort(df_lf_16[N_HF_loc[3]: 2*N_HF_loc[3]][str(loc_no)])
    
    # extract NHF high fidelity samples from each model run and sort them
    output2_2hf = np.sort(df_hf_2[:N_HF_loc[0]][str(loc_no)])
    output4_2hf = np.sort(df_hf_4[:N_HF_loc[1]][str(loc_no)])
    output8_2hf = np.sort(df_hf_8[:N_HF_loc[2]][str(loc_no)])
    output16_2hf = np.sort(df_hf_16[:N_HF_loc[3]][str(loc_no)])

    output2_hf = np.sort(df_hf_2[N_HF_loc[0]: 2*N_HF_loc[0]][str(loc_no)])
    output4_hf = np.sort(df_hf_4[N_HF_loc[1]: 2*N_HF_loc[1]][str(loc_no)])
    output8_hf = np.sort(df_hf_8[N_HF_loc[2]: 2*N_HF_loc[2]][str(loc_no)])
    output16_hf = np.sort(df_hf_16[N_HF_loc[3]: 2*N_HF_loc[3]][str(loc_no)])

    # create inverse samples
    x_ilist_mlmf = []

    for i in range(500000):
        x_i = 0
        u = unif()

        x_i += output4_2hf[np.int(u*len(output4_2hf))] + alpha_list[1][loc_no]*(gamma_2[1][loc_no]*
                    (output4_2lf[np.int(u*len(output4_2lf))]) - outputlist2_exp_nan[0]) 
        
        x_i += output8_2hf[np.int(u*len(output8_2hf))] - output4_hf[np.int(u*len(output4_hf))] + \
            alpha_list[2][loc_no]*((gamma_2[2][loc_no]*(output8_2lf[np.int(u*len(output8_2lf))]) - 
                       output4_lf[np.int(u*len(output4_lf))])-outputlist2_exp_nan[1])
        
        x_i += output16_2hf[np.int(u*len(output16_2hf))] - output8_hf[np.int(u*len(output8_hf))] + \
            alpha_list[3][loc_no]*((gamma_2[3][loc_no]*(output16_2lf[np.int(u*len(output16_2lf))]) - 
                       output8_lf[np.int(u*len(output8_lf))])-outputlist2_exp_nan[2])
       
        
        if x_i > zb[loc_no]:
            # if elevation lower than bed, discard this value
            x_ilist_mlmf.append(x_i)
    
    # plot pdf
    plt.hist(x_ilist_mlmf, bins = 101, histtype=u'step', density = True, label = 'MLMF')
    plt.legend()
    plt.show()    
    
    return x_ilist_mlmf

# apply inverse transform sampling method at each location
x_ilist_mlmf_0 = cdf(0)
x_ilist_mlmf_1 = cdf(1)
x_ilist_mlmf_2 = cdf(2)
x_ilist_mlmf_3 = cdf(3)
x_ilist_mlmf_4 = cdf(4)
x_ilist_mlmf_5 = cdf(5)
x_ilist_mlmf_6 = cdf(6)
x_ilist_mlmf_7 = cdf(7)

hist_mlmf_0, bins_mlmf_0 = np.histogram(np.asarray(x_ilist_mlmf_0), bins = 101, density = True)
hist_mlmf_1, bins_mlmf_1 = np.histogram(np.asarray(x_ilist_mlmf_1), bins = 101, density = True)
hist_mlmf_2, bins_mlmf_2 = np.histogram(np.asarray(x_ilist_mlmf_2), bins = 101, density = True)
hist_mlmf_3, bins_mlmf_3 = np.histogram(np.asarray(x_ilist_mlmf_3), bins = 101, density = True)
hist_mlmf_4, bins_mlmf_4 = np.histogram(np.asarray(x_ilist_mlmf_4), bins = 101, density = True)
hist_mlmf_5, bins_mlmf_5 = np.histogram(np.asarray(x_ilist_mlmf_5), bins = 101, density = True)
hist_mlmf_6, bins_mlmf_6 = np.histogram(np.asarray(x_ilist_mlmf_6), bins = 101, density = True)
hist_mlmf_7, bins_mlmf_7 = np.histogram(np.asarray(x_ilist_mlmf_7), bins = 101, density = True)

## calculate cumulative distributions
cum_hist_mlmf_0 = np.cumsum(hist_mlmf_0)
cum_hist_mlmf_1 = np.cumsum(hist_mlmf_1)
cum_hist_mlmf_2 = np.cumsum(hist_mlmf_2)
cum_hist_mlmf_3 = np.cumsum(hist_mlmf_3)
cum_hist_mlmf_4 = np.cumsum(hist_mlmf_4)
cum_hist_mlmf_5 = np.cumsum(hist_mlmf_5)
cum_hist_mlmf_6 = np.cumsum(hist_mlmf_6)
cum_hist_mlmf_7 = np.cumsum(hist_mlmf_7)

# plot cdfs on two different figures for clarity
fig, ax = plt.subplots()
ax.plot(bins_mlmf_5[1:], cum_hist_mlmf_5/cum_hist_mlmf_5[-1], label = labellist[5], color = colorlist[5])
ax.plot(bins_mlmf_7[1:], cum_hist_mlmf_7/cum_hist_mlmf_7[-1], label = labellist[7], color = colorlist[7])
ax.plot(bins_mlmf_0[1:], cum_hist_mlmf_0/cum_hist_mlmf_0[-1], label = labellist[0], color = colorlist[0])
ax.plot(bins_mlmf_1[1:], cum_hist_mlmf_1/cum_hist_mlmf_1[-1], label = labellist[1], color = colorlist[1])
ax.plot(bins_mlmf_5[1], cum_hist_mlmf_5[1]/cum_hist_mlmf_5[-1], 'k-')

lines = ax.get_lines()
legend1 = pyplot.legend([lines[i] for i in [0, 1, 2, 3]], [labellist[5], labellist[7], labellist[0], labellist[1]], loc =2 )
legend2 = pyplot.legend([lines[i] for i in [4]], ['MLMF'], loc = 4)
ax.add_artist(legend1)
ax.add_artist(legend2)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Output')
ax.set_ylabel(r'$\mathbb{P}(X \leq x)$')
plt.show()

fig, ax = plt.subplots()
ax.plot(bins_mlmf_4[1:], cum_hist_mlmf_4/cum_hist_mlmf_4[-1], label = labellist[4], color = colorlist[4])
ax.plot(bins_mlmf_2[1:], cum_hist_mlmf_2/cum_hist_mlmf_2[-1], label = labellist[2], color = colorlist[2])
ax.plot(bins_mlmf_3[1:], cum_hist_mlmf_3/cum_hist_mlmf_3[-1], label = labellist[3], color = colorlist[3])
ax.plot(bins_mlmf_6[1:], cum_hist_mlmf_6/cum_hist_mlmf_6[-1], label = labellist[6], color = colorlist[6])
ax.plot(bins_mlmf_4[1], cum_hist_mlmf_4[1]/cum_hist_mlmf_4[-1], 'k-')

lines = ax.get_lines()
legend1 = pyplot.legend([lines[i] for i in [0, 1, 2, 3]], [labellist[4], labellist[2], labellist[3], labellist[6]], loc =2 )
legend2 = pyplot.legend([lines[i] for i in [4]], ['MLMF'], loc = 4)
ax.add_artist(legend1)
ax.add_artist(legend2)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Output')
ax.set_ylabel(r'$\mathbb{P}(X \leq x)$')
plt.legend()
plt.show()
