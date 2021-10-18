# Functions for running Monte Carlo algorithm

import time
import numpy as np
import sys
import multiprocessing as mp
import os

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


def monte_carlo(path, qin, qout):
    """
    In this function, the model defined by 'function_name' is run on one level
    """
    os.chdir(path)
    for (function_name, sigma, meshsize) in iter(qin.get, 'stop'):

        L0 = 0
        t0 = time.time()
        outputf_hf = function_name(sigma, meshsize, L0, path)
        t1 = time.time()

        qout.put((outputf_hf, t1-t0))


def _parallel_mc(high_fidelity_fn, meshsize, samples, processes, path_stem, iteration):
    """
    Split the tasks so the algorithm be parallelised and then collect the parallel output
    """

    # putting runs into queues
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    future_res = []
    for i in range(processes):
        path = path_stem + str(i) + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        future_res.append(mp.Process(target = monte_carlo, args = (path, in_queue, out_queue)))
        future_res[-1].start()

    for j in range(iteration):
        sigma = samples[j]
        in_queue.put((high_fidelity_fn, sigma, meshsize))
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

    time_hf = sum([f[1] for f in results])

    return outputf_hf, time_hf
