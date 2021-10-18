"""
 This file runs the MLMF and MLMC algorithm for the carrier-greenspan test case. 
 This test case was first presented in

 Carrier, G.F., Greenspan, H.P., 1958. Water waves of finite amplitude on a sloping beach. 
 Journal of Fluid Mechanics 4, 97–109.
 
 The uncertain parameter here is the slope and the variable of interest is the maximum run-up height.

"""

import numpy as np

import fidelity_fns_carrier as fid_fns
import mlmf_fns_multi as mlmf

prelim_run = True # Runs prelimary samples to determine value of key parameters. 
                  # Should be set to True to run all algorithms
    
opt_sample = True # Should be set to True to run MLMF algorithm, otherwise should be False

opt_hf = False # Should be set to True to run MLMC algorithm with XBeach, otherwise should be False

opt_lf = False # Should be set to True to run MLMC algorithm with SFINCS, otherwise should be False

# number of samples in preliminary run
nhfsamples = int(480)

# meshgrid size
nf = [2**i for i in range(5,10)]

# number of parallel processes
processes = 48

no_out = 1 # number of outputs

# fidelity models
high_fidelity = fid_fns.high_fidelity
low_fidelity = fid_fns.low_fidelity

# function which generates random number for the uncertain slope
def sample_fn():
    slope = -1
    while slope < 0.005:
        slope = np.random.normal(2/50, 0.02)
    return slope

# path where XBeach and SFINCS models are run
xbeach_path = '/rds/general/user/mc4117/ephemeral/MLMC_Code/carrier_runup_mod2/'
sfincs_path = '/rds/general/user/mc4117/home/SFINCS_output/carrier_runup_mod2/'

if prelim_run:
    # preliminary run to determine key MLMF/MLMC parameters

    if opt_lf:
        corr = False
        logfile = 'carrier_ini'
    else:
        corr = True
        logfile = 'carrier_ini'

    r_list, rho_list, gamma_list, variance0, variance1, cost0, cost1, values0_diff_list, values1_diff_list = mlmf.mlmf_init(
       nhfsamples, nf, no_out, processes, high_fidelity, low_fidelity, sample_fn, logfile, xbeach_path, sfincs_path, increase_corr = corr)

    # print values of key parameters
    print('r')
    print(r_list)
    print('rho')
    print(rho_list)
    print('gamma')
    print(gamma_list)
    # print all generated outputs
    print('values0')
    print(values0_diff_list)
    print('values1')
    print(values1_diff_list)

if opt_sample:    
# run MLMF algorithm with optimum number of samples
    
    num_diff = len(r_list[0]) # number of outputs

    N_HF_list = []
    N_LF_list = []

    eps = 1e-3 # set tolerance value

    N_LF_max = [0 for i in range(len(nf))]
    N_HF_max = [0 for i in range(len(nf))]

    for k in range(num_diff):
        # determine optimum number of samples for high fidelity and low fidelity models
        r = [x[k] for x in r_list]
        rho_l = [x[k] for x in rho_list]
        var0 = [x[k] for x in variance0]

        delta = [1 - ((r[i]/(1+r[i]))*rho_l[i]**2) for i in range(len(nf))]

        nf_hf_sum = 0
        for j in range(len(nf)):
            inter = (var0[j]*cost0[j])/(1-rho_l[j]**2)
            nf_hf_sum += np.sqrt(inter)*delta[j]


        N_HF = [(2/(eps**2))*nf_hf_sum*np.sqrt((1-rho_l[i]**2)*var0[i]/cost0[i]) for i in range(len(nf))]
        N_LF = [np.int(np.ceil(r[i]*N_HF[i])) for i in range(len(r))]

        print('N_HF')
        print(N_HF)
        print('N_LF')
        print(N_LF)

        N_HF_list.append(N_HF)
        N_LF_list.append(N_LF)

        N_LF_max = [max(N_LF[i], N_LF_max[i]) for i in range(len(N_LF))]
        N_HF_max = [max(N_HF[i], N_HF_max[i]) for i in range(len(N_HF))]

    samples_calc = [nhfsamples for i in range(len(nf))] # construct an array with number of samples already calculated

    # run MLMF with optimum number of samples
    values0_diff_opt, values1_diff_opt, valueslf_diff_opt = mlmf.mlmf_opt_samples(nf, N_HF_max,
                         N_LF_max, high_fidelity, low_fidelity, samples_calc, sample_fn,
                         values0_diff_list, values1_diff_list, gamma_list,
                         processes, 'carrier_opt_1e3.txt', xbeach_path, sfincs_path, num_diff)    

    #  combime outputs to construct mlmf estimator
    Y_list = mlmf.combine_results(values0_diff_list, values1_diff_list, values0_diff_opt,
                                  values1_diff_opt, valueslf_diff_opt, rho_list,
                                  N_HF_list, N_LF_list, 'carrier_comb_1e3.txt', num_diff)

if opt_hf:
# run MLMC algorithm with XBeach with optimum number of samples    

    num_diff = len(r_list[0]) # number of outputs

    N_HF_list = []

    eps = 1e-3 # set tolerance value

    N_HF_max = [0 for i in range(len(nf))]

    for k in range(num_diff):
        # determine optimum number of samples for high fidelity model
        var0 = [x[k] for x in variance0]

        nf_hf_sum = 0
        for j in range(len(nf)):
            inter = (var0[j]*cost0[j])
            nf_hf_sum += np.sqrt(inter)


        N_HF = [(2/(eps**2))*nf_hf_sum*np.sqrt(var0[i]/cost0[i]) for i in range(len(nf))]

        print('N_HF')
        print(N_HF)

        N_HF_list.append(N_HF)

        N_HF_max = [max(N_HF[i], N_HF_max[i]) for i in range(len(N_HF))]

    samples_calc = [nhfsamples for i in range(len(nf))] # construct an array with number of samples already calculated

    # run MLMC with XBeach (HF) with optimum number of samples
    values0_diff_opt = mlmf.mlmf_opt_samples_single(nf, N_HF_max, N_HF_list,
                         high_fidelity, samples_calc, sample_fn, values0_diff_list, 
                         processes, 'carrier_opt_hf_1e3.txt', xbeach_path, num_diff, fidelity_flag = 'high')    
    
if opt_lf:
# run MLMC algorithm with SFINCS with optimum number of samples    

    num_diff = len(r_list[0]) # number of outputs

    N_LF_list = []

    eps = 1e-3 # set tolerance value

    N_LF_max = [0 for i in range(len(nf))]

    for k in range(num_diff):
        # determine optimum number of samples for low fidelity model
        var1 = [x[k] for x in variance1]

        nf_hf_sum = 0
        for j in range(len(nf)):
            inter = (var1[j]*cost1[j])
            nf_hf_sum += np.sqrt(inter)

        N_LF = [(2/(eps**2))*nf_hf_sum*np.sqrt(var1[i]/cost1[i]) for i in range(len(nf))]

        print('N_LF')
        print(N_LF)

        N_LF_list.append(N_LF)

        N_LF_max = [max(N_LF[i], N_LF_max[i]) for i in range(len(N_LF))]

    samples_calc = [nhfsamples for i in range(len(nf))] # construct an array with number of samples already calculated

    # run MLMC with SFINCS (LF) with optimum number of samples
    values1_diff_opt = mlmf.mlmf_opt_samples_single(nf, N_LF_max, N_LF_list,
                         low_fidelity, samples_calc, sample_fn, values1_diff_list, 
                         processes, 'carrier_opt_lf_1e3.txt', sfincs_path, num_diff, fidelity_flag = 'low')
