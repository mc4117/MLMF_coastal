"""
File defines high and low fidelity models

The models solve the Carrier-Greenspan test problem with an uncertain slope angle.

The low fidelity model is SFINCS and the high fidelity model is XBeach.

"""

import numpy as np
import subprocess
from netCDF4 import Dataset
import os
from shutil import copyfile
from scipy import interpolate

def high_fidelity(sample, meshsize, L0, path):
    # high fidelity model uses XBeach

    if not os.path.exists(path):
        os.makedirs(path)

    # if less than minimum level considered in algorithm, return 0
    if meshsize <= L0:
        return [0]

    os.chdir(path)

    x = np.linspace(0, 150, np.int(meshsize)+1)
    x_file = open('x3.dep', 'w+')
    x_file.write(" ".join(str(i) for i in x))
    x_file.close()
    
    # define bed with uncertain slope angle
    h0 = 5
    bed = np.linspace((-125*sample) + h0, (25*sample) +h0, np.int(meshsize)+1)
    bed_file = open('h3.dep', 'w+')
    bed_file.write(" ".join(str(x) for x in bed))
    bed_file.close()

    # set-up parameter file
    f = open('params.txt', 'w+')

    f.write("nx = ")
    f.write(str(np.int(meshsize)))
    f.write("\n")
    f.write("ny = 0")
    f.write("\n")
    f.write("depfile = h3.dep")
    f.write("\n")
    f.write("vardx = 1")
    f.write("\n")
    f.write("xfile = x3.dep")
    f.write("\n")
    f.write("xori  = 0.")
    f.write("\n")
    f.write("yori     = 0.")
    f.write("\n")
    f.write("alfa     = 0.")
    f.write("\n")
    f.write("posdwn   = -1")
    f.write("\n")
    f.write("break    = 3")
    f.write("\n")
    f.write("Hrms     = 0.0")
    f.write("\n")
    f.write("Tm01     = 1")
    f.write("\n")
    f.write("dir0     = 270")
    f.write("\n")
    f.write("m        = 1024")
    f.write("\n")
    f.write("hmin     = 0.05")
    f.write("\n")
    f.write("Tlong    = 40")
    f.write("\n")
    f.write("gamma    = 0.55")
    f.write("\n")
    f.write("alpha    = 1.")
    f.write("\n")
    f.write("delta    = 0.0")
    f.write("\n")
    f.write("n        = 10.")
    f.write("\n")
    f.write("rho      = 1000")
    f.write("\n")
    f.write("g        = 9.81")
    f.write("\n")
    f.write("thetamin = -100.")
    f.write("\n")
    f.write("thetamax = 100.")
    f.write("\n")
    f.write("dtheta   = 200.")
    f.write("\n")
    f.write("wci      = 0")
    f.write("\n")
    f.write("instat   = 3")
    f.write("\n")
    f.write("nuh      = 0.0")
    f.write("\n")
    f.write("roller   = 0")
    f.write("\n")
    f.write("beta     = 0.1")
    f.write("\n")
    f.write("zs0      = 5")
    f.write("\n")
    f.write("tideloc  = 0")
    f.write("\n")
    f.write("C        = 1E6")
    f.write("\n")
    f.write("eps      = 0.0001")
    f.write("\n")
    f.write("umin     = 0.0")
    f.write("\n")
    f.write("tstart   = 0")
    f.write("\n")
    f.write("tint     = 384")
    f.write("\n")
    f.write("tstop    = 384")
    f.write("\n")
    f.write("tintm    = 384")
    f.write("\n")
    f.write("CFL      = 0.9")
    f.write("\n")
    f.write("front    = 1")
    f.write("\n")
    f.write("freewave = 1")
    f.write("\n")
    f.write("outputformat = netcdf")
    f.write("\n")
    f.write("nglobalvar = 2")
    f.write("\n")
    f.write("zs")
    f.write("\n")
    f.write("zb")
    f.write("\n")
    f.write('nmeanvar = 1')
    f.write("\n")
    f.write("zs")
    f.write("\n")
    f.write("bedfriction  = manning")
    f.write("\n")
    f.write("bedfriccoef  = 0.0")
    f.close()

    if not os.path.exists(os.path.join(path, 'bc')):
        os.makedirs(os.path.join(path, 'bc'))

    # copy relevant boundary conditions files into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Carrier-Greenspan/gen.ezs", os.path.join(path,'bc/gen.ezs'))

    # run model
    subprocess.check_call("/rds/general/user/mc4117/home/trunk/src/xbeach/xbeach > /dev/null", shell=True)

    dataset = Dataset('xboutput.nc')

    # post-process data and extract outputs of interest
    zs_data = dataset.variables['zs_max'][len(dataset.variables['zs_max'])-1].data[0]
    zb_data = dataset.variables['zb'][len(dataset.variables['zb'])-1].data[0]

    diff = [zs_data[i] - zb_data[i] for i in range(len(zs_data))]

    return [zs_data[np.nonzero(diff)][-1]]


def low_fidelity(sample, meshsize, L0, path):
    # low fidelity model uses SFINCS

    if not os.path.exists(path):
        os.makedirs(path)

    # if less than minimum level considered in algorithm, return 0
    if meshsize <= L0:
        return [0, 0]

    os.chdir(path)

    dx = 150/meshsize
    # define bed
    h0 = 5
    bed = np.linspace((-125*sample) + h0, (25*sample) +h0, np.int(meshsize)+1)
    bed_file = open('h3.dep', 'w+')
    bed_file.write(" ".join(str(x) for x in bed))
    bed_file.close()
    
    msk = [2]
    for i in range(1, np.int(meshsize)):
        msk.append(1)
        
    msk_file = open('sfincs.msk', 'w+')
    msk_file.write(" ".join(str(i) for i in msk))
    msk_file.close()    
    
    cwd = os.getcwd()
    
    # set-up parameter file
    f = open('sfincs.inp', 'w+')
    f.write("mmax = ")
    f.write(str(np.int(meshsize)))    
    f.write("\n")
    f.write("nmax = 1")    
    f.write("\n")
    f.write("dy = 1")
    f.write("\n")
    f.write("dx = ")
    f.write(str(dx))
    f.write("\n")
    f.write("x0  = 0")
    f.write("\n") 
    f.write("y0  = 0")
    f.write("\n")
    f.write("rotation  = 0")
    f.write("\n") 
    f.write("tref = 20180101 000000")
    f.write("\n") 
    f.write("tstart         = 20180101 000000")
    f.write("\n") 
    f.write("tstop          = 20180101 000624")
    f.write("\n") 
    f.write("dtout = 384")
    f.write("\n") 
    f.write("dthisout = 384")
    f.write("\n")
    f.write("dtmaxout = 384")
    f.write("\n")
    f.write("dtwnd          = 1800")
    f.write("\n") 
    f.write("alpha          = 0.75")
    f.write("\n")
    f.write("manning        = 0")
    f.write("\n")
    f.write("zsini          = 5")
    f.write("\n") 
    f.write("qinf           = 0")
    f.write("\n")
    f.write("depfile        = h3.dep")
    f.write("\n")     
    f.write("mskfile        = sfincs.msk")
    f.write("\n")
    f.write("bndfile        = sfincs.bnd")
    f.write("\n")    
    f.write("bzsfile        = sfincs.bzs")
    f.write("\n")
    f.write("bzifile        = sfincs.bzi")
    f.write("\n")
    f.write("inputformat    = asc")
    f.write("\n") 
    f.write("outputformat   = net")
    f.write("\n") 
    f.write("huthresh = 0.0001")
    f.write("\n")     
    f.write("advection      = 0") 
    f.write("\n") 
    f.write("bndtype        = 1")
    f.write("\n") 
    f.write("tspinup = 0")  

    f.close()
    
    cwd = os.getcwd()

    # # copy relevant boundary conditions files into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Carrier-Greenspan/sfincs.bnd", os.path.join(cwd,'sfincs.bnd'))

    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Carrier-Greenspan/sfincs.bzs", os.path.join(cwd,'sfincs.bzs'))

    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Carrier-Greenspan/sfincs.bzi", os.path.join(cwd,'sfincs.bzi'))

    # run model
    subprocess.check_call("singularity run /rds/general/user/mc4117/home/SFINCS_MLMF/sfincs-cpu_latest.sif > /dev/null", shell=True)

    dataset = Dataset('sfincs_map.nc')
    
    # post-process data and extract outputs of interest
    zs_data = dataset['zsmax'][len(dataset['zsmax'])-1].data[dataset['zsmax'][0].mask==False]

    return [zs_data[-1]]
    
    
