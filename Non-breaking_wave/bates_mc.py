"""
 This file runs the Monte Carlo algorithm using XBeach for the non-breaking wave test case. 
 
 Note that this file must be run multiple times to generate a sufficient number of outputs 
 to verify the MLMF and MLMC results.
 
 This test case was first presented in

 Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010.  A simple inertial formulation of the shallow water
 equations for efficient two-dimensional flood inundation modelling. Journal of Hydrology 387, 33–45.
 
 The uncertain parameter here is the Manning friction coefficient and 
 the variable of interest is the water elevation height at four different locations.
"""

import numpy as np
import pandas as pd
import time
import datetime

import fidelity_fns_bates as fid_fns
import mc_fns as mc

monte_carlo = True

# number of parallel processes
processes = 48

no_out = 4 # number of outputs

# fidelity models
high_fidelity = fid_fns.high_fidelity
low_fidelity = None 

# function which generates random number for uncertain Manning friction coefficient
def sample_fn():
    return np.random.normal(0.03, 0.01)

path = '/rds/general/user/mc4117/ephemeral/MLMC_Code/bates_test_mc/0/'

if monte_carlo:
    # Basic monte carlo run for 480 samples with high fidelity model
    hf_samples = [sample_fn() for i in range(480)]
    meshsize = 2**10

    outputf_hf, time_hf = mc._parallel_mc(high_fidelity, meshsize, hf_samples, processes, path, iteration = len(hf_samples))

    # extract and record output
    output_hf_0 = [x[0] for x in outputf_hf]
    output_hf_1	= [x[1]	for x in outputf_hf]
    output_hf_2	= [x[2]	for x in outputf_hf]
    output_hf_3	= [x[3]	for x in outputf_hf]
    output_hf_4	= [x[4]	for x in outputf_hf]

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    pd.DataFrame(output_hf_0).to_csv('output_hf_0_' + st + '.csv')
    pd.DataFrame(output_hf_1).to_csv('output_hf_1_' + st + '.csv')
    pd.DataFrame(output_hf_2).to_csv('output_hf_2_' + st + '.csv')
    pd.DataFrame(output_hf_3).to_csv('output_hf_3_' + st + '.csv')
    pd.DataFrame(output_hf_4).to_csv('output_hf_4_' + st + '.csv')

    logfile_name = 'bates_mc_' + st + '.txt'

    mc.write(logfile_name, "No of samples: ")
    mc.write(logfile_name, str(len(hf_samples)))
    mc.write(logfile_name, "\n")
    mc.write(logfile_name, "Expected value: ")
    mc.write(logfile_name, str(np.mean(output_hf_0)))
    mc.write(logfile_name, ", ")
    mc.write(logfile_name, str(np.mean(output_hf_1)))
    mc.write(logfile_name, ", ")
    mc.write(logfile_name, str(np.mean(output_hf_2)))
    mc.write(logfile_name, ", ")
    mc.write(logfile_name, str(np.mean(output_hf_3)))
    mc.write(logfile_name, ", ")
    mc.write(logfile_name, str(np.mean(output_hf_4)))
    mc.write(logfile_name, "\n")
    mc.write(logfile_name, "Total time: ")
    mc.write(logfile_name, str(time_hf))
