"""
 This file runs the Monte Carlo algorithm using XBeach for the non-breaking wave test case. 
 
 Note that this file must be run multiple times to generate a sufficient number of outputs 
 to verify the MLMF and MLMC results.
 
 This test case was first presented in

 Carrier, G.F., Greenspan, H.P., 1958. Water waves of finite amplitude on a sloping beach. Journal of Fluid Mechanics 4, 97–109.
 
 The uncertain parameter here is the slope and the variable of interest is the maximum run-up height.
 
"""

import numpy as np
import pandas as pd
import time
import datetime

import fidelity_fns_carrier as fid_fns
import mc_fns as mc

monte_carlo = True

# number of parallel processes
processes = 48

no_out = 1 # number of outputs

# fidelity models
high_fidelity = fid_fns.high_fidelity
low_fidelity = None

# function which generates random number for uncertain slope angle
def sample_fn():
    slope = -1
    while slope < 0.005:
        slope = np.random.normal(2/50, 0.02)
    return slope

path = '/rds/general/user/mc4117/ephemeral/MLMC_Code/carrier_runup_mc/0/'

if monte_carlo:
    # Basic monte carlo run for 2400 samples with high fidelity model
    hf_samples = [sample_fn() for i in range(2400)]
    meshsize = 2**9

    outputf_hf, time_hf = mc._parallel_mc(high_fidelity, meshsize, hf_samples, processes, path, iteration = len(hf_samples))

    # extract and record output
    output_hf_0 = [x[0] for x in outputf_hf]

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    pd.DataFrame(output_hf_0).to_csv('fix_output_hf_0_' + st + '.csv')

    logfile_name = 'carrier_fix_mc_' + st+ '.txt'

    mc.write(logfile_name, "No of samples: ")
    mc.write(logfile_name, str(len(hf_samples)))
    mc.write(logfile_name, "\n")
    mc.write(logfile_name, "Expected value: ")
    mc.write(logfile_name, str(np.mean(output_hf_0)))
    mc.write(logfile_name, "\n")
    mc.write(logfile_name, "Total time: ")
    mc.write(logfile_name, str(time_hf))
