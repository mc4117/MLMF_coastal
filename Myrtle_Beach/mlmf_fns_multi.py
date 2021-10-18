"""
Python wrapper to run MLMF algorithm and MLMC algorithm
"""

import os
import time
import numpy as np
import sys
import multiprocessing as mp
import pandas as pd

from scipy.stats import kurtosis


def write(logfile, msg):
    """
    Write to both sys.stdout and to a logfile.
    """
    logfile = open(logfile, 'a+')
    logfile.write(msg)
    sys.stdout.write(msg)
    sys.stdout.flush()
    logfile.close()

def multilevel(hf_path, lf_path, qin, qout):
    """
    Runs high fidelity and low fidelity model on level l and level (l-1) using the same random number
    """
    for (low_fid_fn, high_fid_fn, sigma, meshsize, L0) in iter(qin.get, 'stop'):

        t0 = time.time()
        if high_fid_fn is not None:
            os.chdir(hf_path)
            outputf_hf = high_fid_fn(sigma, meshsize, L0, hf_path)
            outputc_hf = high_fid_fn(sigma, meshsize/2, L0, hf_path)
        else:
            outputf_hf = None
            outputc_hf = None

        t1 = time.time()
        if low_fid_fn is not None:
            os.chdir(lf_path)
            outputf_lf = low_fid_fn(sigma, meshsize, L0, lf_path)
            outputc_lf = low_fid_fn(sigma, meshsize/2, L0, lf_path)
        else:
            outputf_lf = [None for i in range(len(outputf_hf))]
            outputc_lf = [None for i in	range(len(outputf_hf))]
        t2 = time.time()

        qout.put((outputf_hf, outputc_hf, outputf_lf, outputc_lf, t1-t0, t2-t1))


def _parallel_mlmf(low_fidelity_fn, high_fidelity_fn, samples, meshsize, L0, processes, iteration, hf_path_stem, lf_path_stem):
    """
    Split the tasks so the algorithm be parallelised and then collect the parallel output
    """

    # putting runs into queues
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    future_res = []
    for i in range(processes):
        hf_path = hf_path_stem + str(i) + '/'
        if not os.path.exists(hf_path):
            os.makedirs(hf_path)
        if lf_path_stem is not None:
            lf_path = lf_path_stem + str(i) + '/'
            if not os.path.exists(lf_path):
                os.makedirs(lf_path)
        else:
            lf_path = None

        future_res.append(mp.Process(target = multilevel, args = (hf_path, lf_path, in_queue, out_queue)))
        future_res[-1].start()

    for j in range(iteration):
        sigma = samples[j]
        in_queue.put((low_fidelity_fn, high_fidelity_fn, sigma, meshsize, L0))
    # send stop signals
    for i in range(processes):
        in_queue.put('stop')

    # collect output
    results = []
    for i in range(iteration):
        if (i+1)%1000 == 0:
            print(i)
        results.append(out_queue.get())

    outputf_hf = [f[0] for f in results]
    outputc_hf = [f[1] for f in results]

    outputf_lf = [f[2] for f in results]
    outputc_lf = [f[3] for f in results]

    time_hf = sum([f[4] for f in results])
    time_lf = sum([f[5] for f in results])
    
    tst = np.int(time.time())
    
    # record output in csv files (then used to construct CDFs)    
    pd.DataFrame(outputf_lf).to_csv('outputf_lf' + str(meshsize) + '_'+ str(tst) + '.csv')
    pd.DataFrame(outputc_lf).to_csv('outputc_lf' + str(meshsize) + '_'+ str(tst) + '.csv')  
    
    pd.DataFrame(outputf_hf).to_csv('outputf_hf' + str(meshsize) + '_'+ str(tst) + '.csv')
    pd.DataFrame(outputc_hf).to_csv('outputc_hf' + str(meshsize) + '_'+ str(tst) + '.csv')    
    
    timefile = 'time_' + str(tst) + '.txt'
    write(timefile, "Time high fidelity: " + str(time_hf))
    write(timefile, "\n")
    write(timefile, "Time low fidelity: " + str(time_lf))

    return outputf_hf, outputc_hf, outputf_lf, outputc_lf, time_hf, time_lf

def mlmf_init(nhfsamples, nf, no_out, no_parallel_processes, high_fidelity, low_fidelity, sample_fn, logfile, hf_path, lf_path, L0 = None, increase_corr = True):

    """
    Multilevel Multifidelity Monte Carlo routine for preliminary run. Also checks the viability of MLMC for each model fidelity.

    nhfsamples: initial number of samples for MLMF calculations
    nf: array of meshgrid sizes being considered
    no_out: number of outputs
    no_parallel_processes: number of different parallel runs of the model
    high_fidelity: function which runs the high fidelity model
    low_fidelity: function which runs the low fidelity model
    sample_fn: function used to generate random number
    logfile: root name of file where outputs are stored
    hf_path: directory where high fidelity model is run
    lf_path: directory where low fidelity model is run
    L0: minimum level
    increase_corr: flag indicating whether to maximize the correlation between the two models
    """

    # initialize lists
    rho_list = []
    rho_no_corr_list = []
    r_list = []
    gamma_list = []
    omega_list = []
    variance0 = []
    variance1 = []
    cost0 = []
    cost1 = []
    values0_diff_list = []
    values1_diff_list = []

    for k in range(no_out):
        # initialize logfiles
        logfile_hf = logfile + "_" + str(k) + "_hf.txt"
        logfile_lf = logfile + "_" + str(k) + "_lf.txt"

        write(logfile_hf, "\n")
        write(logfile_hf, "**********************************************************\n")
        write(logfile_hf, "*** Convergence tests, kurtosis, telescoping sum check for high fidelity ***\n")
        write(logfile_hf, "**********************************************************\n")
        write(logfile_hf, "\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)    cost")
        write(logfile_hf, "    kurtosis     check \n-------------------------")
        write(logfile_hf, "--------------------------------------------------\n")

        write(logfile_lf, "\n")
        write(logfile_lf, "**********************************************************\n")
        write(logfile_lf, "*** Convergence tests, kurtosis, telescoping sum check for low fidelity ***\n")
        write(logfile_lf, "**********************************************************\n")
        write(logfile_lf, "\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)    cost")
        write(logfile_lf, "    kurtosis     check \n-------------------------")
        write(logfile_lf, "--------------------------------------------------\n")

    for j in nf:

        hf_samples = [sample_fn() for _ in range(nhfsamples)]

        meshsize = j

        # run the preliminary MLMC algorithm for the high fidelity model and low fidelity model
        if L0 is None:
            L0 = min(nf)/2 # if L0 none then assume the minimum level is the coarsest mesh in the simulation
        
        values0_l, values0_l_1, values1_l, values1_l_1, time_hf, time_lf = _parallel_mlmf(low_fidelity, high_fidelity, hf_samples, meshsize, L0,
                                                                                        no_parallel_processes, nhfsamples, hf_path, lf_path)
        rho_k_list = []
        rho_no_corr_k = []
        r_k_list = []
        gamma_k_list = []
        variance0_k_list = []
        variance1_k_list = []
        values0_diff_k_list = []
        values1_diff_k_list = []

        for k in range(len(values0_l[0])):

            # calculate P_l - P_(l-1) for high fidelity model
            values0_l_k = [x[k] for x in values0_l]
            values0_l_1_k = [x[k] for x in values0_l_1]

            values0_diff = [values0_l_k[i] - values0_l_1_k[i] for i in range(len(values0_l_k))]

            # verification check and kurtosis calculation for MLMC with high fidelity model
            if meshsize == min(nf):
                check = 0
                kurt = 0.0
            else:
                check = abs(np.mean(values0_diff) + np.mean(values0_l_1_k) -
                        np.mean(values0_l_k))/(3.0*(np.sqrt(np.var(values0_diff)) +
                                                  np.sqrt(np.var(values0_l_1_k)) +
                                                  np.sqrt(np.var(values0_l_k))))
                kurt = kurtosis(values0_diff)

            logfile_hf = logfile + "_" + str(k) + "_hf.txt"
            logfile_lf = logfile + "_" + str(k) + "_lf.txt"

            write(logfile_hf, "%2d   %8.4e  %8.4e  %8.4e  %8.4e  %8.4e  %8.4e %8.4e \n" %
               (j, np.mean(values0_diff), np.mean(values0_l_k), np.var(values0_diff), np.var(values0_l_k), time_hf, kurt, check))

            if low_fidelity is not None:
                if meshsize > min(nf):
                    values1_l_1_k = [x[k] for x in values1_l_1]
                else:
                    values1_l_1_k = [x[k] for x in values1_l_1]
                values1_l_k = [x[k] for x in values1_l]

                # calculate gamma which optimizes the correlation between the high and low fidelity models
                if meshsize > min(nf) and increase_corr:
                    cov_LF = np.cov(values1_l_k, values1_l_1_k, ddof = 0)
                    cov_HF_l_1 = np.cov(values0_diff, values1_l_1_k, ddof = 0)
                    cov_HF_l = np.cov(values0_diff, values1_l_k, ddof = 0)

                    gamma = ((cov_HF_l_1[0,1]*cov_LF[0,1])-(cov_LF[1,1]*cov_HF_l[0,1]))/((cov_LF[0,0]*cov_HF_l_1[0,1])-(cov_HF_l[0,1]*cov_LF[0,1]))
                else:
                    # if increased correlation is not desired then gamma is set to 1
                    gamma = 1

                # reweight the fine values from the low fidelity model in order to maximize correlation
                values1_diff_circ = [gamma*values1_l_k[i] - values1_l_1_k[i] for i in range(len(values1_l_k))]
                values1_no_corr = [values1_l_k[i] - values1_l_1_k[i] for i in range(len(values1_l_k))]

                # verification check and kurtosis calculation for MLMC with low fidelity model
                if meshsize == min(nf):
                    check = 0
                    kurt = 0.0
                else:
                    check = abs(np.mean(values1_diff_circ) + np.mean(values1_l_1_k) -
                        gamma*np.mean(values1_l_k))/(3.0*(np.sqrt(np.var(values1_diff_circ)) +
                                                  np.sqrt(np.var(values1_l_1_k)) +
                                                  gamma*np.sqrt(np.var(values1_l_k))))
                    kurt = kurtosis(values1_diff_circ)

                write(logfile_lf, "%2d   %8.4e  %8.4e  %8.4e  %8.4e  %8.4e  %8.4e %8.4e \n" %
                 (j, np.mean(values1_diff_circ), gamma*np.mean(values1_l_k), np.var(values1_diff_circ), (gamma**2)*np.var(values1_l_k), time_lf, kurt, check))

                cov_circ = np.cov(values0_diff, values1_diff_circ, ddof = 0)

                # calculate new correlation between models
                rho_circ = cov_circ[0, 1]/np.sqrt(np.var(values0_diff)*np.var(values1_diff_circ))
                rho_no_corr = np.cov(values0_diff, values1_no_corr, ddof = 0)[0,1]/np.sqrt(np.var(values0_diff)*np.var(values1_no_corr))
                # computation cost ratio
                omega = time_hf/time_lf
                # calculate optimum ratio for N_LF (see later function)
                r = -1 + np.sqrt(omega*rho_circ**2/(1-rho_circ**2))

                values1_diff_k_list.append(values1_diff_circ)
                variance1_k_list.append(np.var(values1_diff_circ))
            else:
                omega = 0
                r = 0
                gamma = 0
                rho_circ = None
                values1_diff_k_list = None
                variance1_k_list = None
        
            r_k_list.append(r)
            rho_k_list.append(rho_circ)
            rho_no_corr_k.append(rho_no_corr)
            gamma_k_list.append(gamma)
            variance0_k_list.append(np.var(values0_diff))
            values0_diff_k_list.append(values0_diff)

        # record outputs
        r_list.append(r_k_list)
        rho_list.append(rho_k_list)
        rho_no_corr_list.append(rho_no_corr_k)
        omega_list.append(omega)
        gamma_list.append(gamma_k_list)
        variance0.append(variance0_k_list)
        variance1.append(variance1_k_list)
        cost0.append(time_hf)
        cost1.append(time_lf)
        values0_diff_list.append(values0_diff_k_list)
        values1_diff_list.append(values1_diff_k_list)

        print('omega')
        print(omega_list)
        print('rho')
        print(rho_no_corr_list)

    return r_list, rho_list, gamma_list, variance0, variance1, cost0, cost1, values0_diff_list, values1_diff_list


def mlmf_opt_samples(nf, N_HF, N_LF, high_fidelity, low_fidelity, samples_calc, sample_fn,
                 values0_diff_list, values1_diff_list, gamma_list, no_parallel_processes,
                 logfile_name, hf_path, lf_path, no_out, L0 = None):

    """
    MLMF routine for optimum number of samples.

    nf: array of meshgrid sizes being considered
    N_HF: optimum number of samples for high fidelity model
    N_LF: optimum number of extra samples required for low fidelity model
    high_fidelity: function which runs the high fidelity model
    low_fidelity: function which runs the low fidelity model
    samples_calc: initial number of samples for MLMF calculations
    sample_fn: function used to generate random number
    values0_diff_list: high fidelity model output from prelim run
    values1_diff_list: low fidelity model output for prelim run
    gamma_list: optimum value to optimize correlation between models
    no_parallel_processes: number of different parallel runs of the model
    logfile_name: root name of file where outputs are stored
    hf_path: directory where high fidelity model is run
    lf_path: directory where low_fidelity model is run
    no_out: number of outputs
    L0: minimum level
    """

    # record the optimum number of results for the high and low fidelity models
    write(logfile_name, 'N_HF max')
    write(logfile_name, "\n")
    write(logfile_name, str([np.int(np.ceil(i)) for i in N_HF]))
    write(logfile_name, "\n")
    write(logfile_name, 'N_LF max')
    write(logfile_name, "\n")
    write(logfile_name, str(N_LF))
    write(logfile_name, "\n")

    total_time = 0

    values0_diff_opt = []
    values1_diff_opt = []
    valueslf_diff_opt = []

    for j in range(len(nf)):

        write(logfile_name, str(j))

        if np.ceil(N_HF[j]) <= samples_calc[j]:
            # if the optimum number is less than the number of results used in the preliminary run,
            # take the optimum number of results from the preliminary run rather than re-running the models
            n_samples = np.int(np.ceil(N_HF[j]))

            values0_diff_k = []
            values1_diff_k = []
            
            if n_samples > 0:
                for k in range(no_out):
                    values0_diff_k.append(values0_diff_list[j][k][:n_samples])
                    values1_diff_k.append(values1_diff_list[j][k][:n_samples])
            else:
                print('here')
                for k in range(no_out):
                    values0_diff_k.append([])
                    values1_diff_k.append([])

            values0_diff_opt.append(values0_diff_k)
            values1_diff_opt.append(values1_diff_k)
            time_hf = 0; time_lf = 0
        else:
            # if optimum number is more than the number of results used in the preliminary run
            # use all the results from the preliminary run and then generate the rest of the results to
            # reach the optimum number

            diff_nj = np.int(np.ceil(N_HF[j]) - samples_calc[j])

            hf_samples = [sample_fn() for _ in range(diff_nj)]

            meshsize = nf[j]

            # high fidelity and low fidelity
            if L0 is None:
                L0 = min(nf)/2 # if L0 none then assume the minimum level is the coarsest mesh in the simulation
            
            values0_l, values0_l_1, values1_l, values1_l_1, time_hf, time_lf = _parallel_mlmf(low_fidelity, high_fidelity, hf_samples, meshsize, L0,
                                                                                        no_parallel_processes, diff_nj, hf_path, lf_path)

            values0_diff_k = []
            values1_diff_k = []

            for k in range(no_out):
                values0_l_k = [x[k] for x in values0_l]
                values0_l_1_k = [x[k] for x in values0_l_1]
                values0_diff_int_k = [values0_l_k[i] - values0_l_1_k[i] for i in range(len(values0_l_k))]

                values1_l_k = [x[k] for x in values1_l]

                if meshsize > nf[0]:
                    values1_l_1_k = [x[k] for x in values1_l_1]
                else:
                    values1_l_1_k = [x[k] for x in values1_l_1]
                gamma = [x[k] for x in gamma_list]
                values1_diff_int_k = [gamma[j]*values1_l_k[i] - values1_l_1_k[i] for i in range(len(values1_l_k))]

                # combine the preliminary run results with the extra results generated
                values0_diff = np.concatenate([values0_diff_list[j][k], values0_diff_int_k])
                values1_diff = np.concatenate([values1_diff_list[j][k], values1_diff_int_k])

                values0_diff_k.append(values0_diff)
                values1_diff_k.append(values1_diff)

            values0_diff_opt.append(values0_diff_k)
            values1_diff_opt.append(values1_diff_k)

            n_samples = samples_calc[j]

        # next we generate the extra low fidelity results as required by the MLMF algorithm
        if np.ceil(N_LF[j]) <= samples_calc[j] - n_samples:
            # if there are still results left over from the preliminary runs use them here
            newlf_samples = np.int(np.ceil(N_LF[j]))
            values_lf_diff_k = []
            for k in range(no_out):
                values_lf_diff = values1_diff_list[j][k][n_samples:n_samples+newlf_samples]
                values_lf_diff_k.append(values_lf_diff)
            valueslf_diff_opt.append(values_lf_diff_k)
            time_lf_2 = 0
        else:
            # if there are not enough results from the preliminary run left over then generate more results
            # until the optimum number is reached

            meshsize = nf[j]
            diff_lf_nj = np.int(np.ceil(N_LF[j]-(samples_calc[j]-n_samples)))
            lf_samples = [sample_fn() for _ in range(diff_lf_nj)]

            # for this section we only need to generate results from the low fidelity model
            if L0 is None:
                L0 = min(nf)/2 # if L0 none then assume the minimum level is the coarsest mesh in the simulation
            tmp0, tmp1, values1_lf_l, values1_lf_l_1, time_hf_2, time_lf_2 = _parallel_mlmf(low_fidelity, None, lf_samples, meshsize, L0,
                                                                                        no_parallel_processes, diff_lf_nj, hf_path, lf_path)

            values_lf_diff_k = []

            for k in range(no_out):
                values1_lf_l_k = [x[k] for x in values1_lf_l]
                if meshsize > nf[0]:
                    values1_lf_l_1_k = [x[k] for x in values1_lf_l_1]
                else:
                    values1_lf_l_1_k = [x[k] for x in values1_lf_l_1]                

                gamma = [x[k] for x in gamma_list]
                values_lf_diff_int = [gamma[j]*values1_lf_l_k[i] - values1_lf_l_1_k[i] for i in range(len(values1_lf_l_k))]

                # combine the results from the preliminary run and the new results that have been generated
                values_lf_diff = np.concatenate([values_lf_diff_int, values1_diff_list[j][k][n_samples:]])
                values_lf_diff_k.append(values_lf_diff)

            valueslf_diff_opt.append(values_lf_diff_k)

        write(logfile_name, "timelf: ")
        write(logfile_name, str(time_lf))
        write(logfile_name, " ")
        write(logfile_name, "timehf: ")
        write(logfile_name, str(time_hf))
        write(logfile_name, " ")
        write(logfile_name, "timelf_2: ")
        write(logfile_name, str(time_lf_2))
        write(logfile_name, "\n")

        total_time += time_lf + time_hf + time_lf_2

    write(logfile_name, "Total time: ")
    write(logfile_name, str(total_time))

    return values0_diff_opt, values1_diff_opt, valueslf_diff_opt


def mlmf_opt_samples_single(nf, N_HF, N_HF_list, high_fidelity, samples_calc, sample_fn,
                 values0_diff_list, no_parallel_processes,
                 logfile_name, hf_path, no_out, fidelity_flag):

    """
    MLMC routine for optimum number of samples for single fidelity function.

    nf: array of meshgrid sizes being considered
    N_HF: optimum number of samples for high fidelity model (max over all outputs)
    N_HF_list: optimum number of samples for all outputs
    high_fidelity: function which runs the high fidelity model
    samples_calc: initial number of samples for MLMF calculations
    sample_fn: function used to generate random number
    values0_diff_list: high fidelity model output from prelim run
    no_parallel_processes: number of different parallel runs of the model
    logfile_name: root name of file where outputs are stored
    hf_path: directory where high fidelity model is run
    no_out: number of outputs
    fidelity_flag: flag which sets whether fidelity function is high or low
    """

    # record the optimum number of results for the model
    write(logfile_name, 'N_HF max')
    write(logfile_name, "\n")
    write(logfile_name, str([np.int(np.ceil(i)) for i in N_HF]))
    write(logfile_name, "\n")

    total_time = 0

    values0_diff_opt = []


    for j in range(len(nf)):

        write(logfile_name, str(j))

        if np.ceil(N_HF[j]) <= samples_calc[j]:
            # if the optimum number is less than the number of results used in the preliminary run,
            # take the optimum number of results from the preliminary run rather than re-running the models
            n_samples = np.int(np.ceil(N_HF[j]))

            values0_diff_k = []

            for k in range(no_out):
                values0_diff_k.append(values0_diff_list[j][k][:n_samples])

            values0_diff_opt.append(values0_diff_k)
            time_hf = 0
        else:
            # if optimum number is more than the number of results used in the preliminary run
            # use all the results from the preliminary run and then generate the rest of the results to
            # reach the optimum number

            diff_nj = np.int(np.ceil(N_HF[j]) - samples_calc[j])

            hf_samples = [sample_fn() for _ in range(diff_nj)]

            meshsize = nf[j]

            # high fidelity and low fidelity
            values0_l, values0_l_1, tmp_l, tmp_l_1, time_hf, tmp_time_lf = _parallel_mlmf(None, high_fidelity, hf_samples, meshsize, min(nf)/2,
                                                                                        no_parallel_processes, diff_nj, hf_path, None)

            values0_diff_k = []

            for k in range(no_out):
                if fidelity_flag == 'low':
                     values0_l_k = [x[0][k] for x in values0_l]
                     if meshsize > nf[0]:
                         values0_l_1_k = [x[0][k] for x in values0_l_1]
                     else:
                         values0_l_1_k = [x[k] for x in values0_l_1]
                elif fidelity_flag == 'high':
                     values0_l_k = [x[k] for x in values0_l]
                     values0_l_1_k = [x[k] for x in values0_l_1]
                else:
                     print('Error: must assign fidelity flag')
                     return None
     
                values0_diff_int_k = [values0_l_k[i] - values0_l_1_k[i] for i in range(len(values0_l_k))]

                # combine the preliminary run results with the extra results generated
                values0_diff = np.concatenate([values0_diff_list[j][k], values0_diff_int_k])

                values0_diff_k.append(values0_diff)

            values0_diff_opt.append(values0_diff_k)

            n_samples = samples_calc[j]

        write(logfile_name, "timehf: ")
        write(logfile_name, str(time_hf))
        write(logfile_name, " ")
        total_time += time_hf

    write(logfile_name, "Total time: ")
    write(logfile_name, str(total_time))

    Y_list = []
    for k in range(no_out):

        Y_l = 0
        Y_k_list = []

        N_HF = N_HF_list[k]

        write(logfile_name, 'N_HF')
        write(logfile_name, "\n")
        write(logfile_name, str([np.int(np.ceil(i)) for i in N_HF]))

        values0_diff = [x[k] for x in values0_diff_opt]

        #MLMF estimator
        for j in range(len(N_HF)):
            N_HF_j = np.int(np.ceil(N_HF[j]))
            Y_l += np.mean(values0_diff[j][:N_HF_j])
            Y_k_list.append(Y_l)

        write(logfile_name, "\n")
        write(logfile_name, 'Y_l')
        write(logfile_name, "\n")
        write(logfile_name, str(Y_k_list))
        write(logfile_name, "\n")

        # record expected value
        write(logfile_name, 'Expected value')
        write(logfile_name, "\n")
        write(logfile_name, str(Y_k_list[-1]))
        write(logfile_name, "\n")

        Y_list.append(Y_k_list)

    return Y_list

def combine_results(values0_diff_orig, values1_diff_orig, values0_diff_opt, values1_diff_opt,
                    valueslf_diff_opt, rho_list, N_HF_list, N_LF_list, logfile_name, no_out):

    """
    Combine output of mlmf_opt_sample to create MLMF estimator
    
    values0_diff_orig: high fidelity model output from prelim run
    values1_diff_orig: low fidelity model output from prelim run
    values0_diff_opt: high fidelity model output from optimum number of sample run
    values1_diff_opt: low fidelity model output from optimum number of sample run (N_HF)
    valueslf_diff_opt: extra low fidelity model output 
                       for expectation estimator from optimum number of sample run (N_LF)
    rho_list: correlation between each model
    N_HF_list: number of optimum high fidelity samples
    N_LF_list: number of optimum extra low fidelity samples
    logfile_name: root name of file where outputs are stored
    no_out: number of outputs
    """

    Y_list = []
    for k in range(no_out):

        Y_l = 0
        Y_k_list = []

        N_HF = N_HF_list[k]
        N_LF = N_LF_list[k]

        write(logfile_name, 'N_HF')
        write(logfile_name, "\n")
        write(logfile_name, str([np.int(np.ceil(i)) for i in N_HF]))
        write(logfile_name, "\n")

        write(logfile_name, 'N_LF')
        write(logfile_name, "\n")
        write(logfile_name, str([np.int(np.ceil(i)) for i in N_LF]))
        write(logfile_name, "\n")

        rho = [x[k] for x in rho_list]

        values0_diff = [x[k] for x in values0_diff_opt]
        values1_diff = [x[k] for x in values1_diff_opt]

        # calculate the variances
        variance0 = [np.var(varlist) for varlist in [x[k] for x in values0_diff_orig]]
        variance1 = [np.var(varlist) for varlist in [x[k] for x in values1_diff_orig]]

        valueslf_diff = [x[k] for x in valueslf_diff_opt]

        # optimum value to combine low fidelity and high fidelity model
        alpha = [-rho[j]*np.sqrt(variance0[j]/variance1[j]) for j in range(len(rho))]

        #MLMF estimator
        for j in range(len(rho)):
            N_HF_j = np.int(np.ceil(N_HF[j]))
            N_LF_j = np.int(np.ceil(N_LF[j]))
            if N_LF_j < 0:
                new_lf = N_HF_j + N_LF_j
                print(new_lf)
                if new_lf > 0:
                    Y_l += np.mean(values0_diff[j][:N_HF_j]) + alpha[j]*(np.mean(values1_diff[j][:N_HF_j]) - np.mean(values1_diff[j][:new_lf]))
                else:
                    print('warning')
                    Y_l += np.mean(values0_diff[j][:N_HF_j])+ alpha[j]*(np.mean(values1_diff[j][:N_HF_j]))
            else:
                Y_l += np.mean(values0_diff[j][:N_HF_j]) + alpha[j]*(np.mean(values1_diff[j][:N_HF_j]) - np.concatenate([values1_diff[j][:N_HF_j], valueslf_diff[j][:N_LF_j]]).mean())
            Y_k_list.append(Y_l)

        write(logfile_name, "\n")
        write(logfile_name, 'Y_l')
        write(logfile_name, "\n")
        write(logfile_name, str(Y_k_list))
        write(logfile_name, "\n")

        # record expected value
        write(logfile_name, 'Expected value')
        write(logfile_name, "\n")
        write(logfile_name, str(Y_k_list[-1]))
        write(logfile_name, "\n")

        Y_list.append(Y_k_list)

    return Y_list
