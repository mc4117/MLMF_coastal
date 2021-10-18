"""
File defines high and low fidelity models

The models solve the Bates test problem with an uncertain Manning friction coefficient.

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
        return [0 for i in [1000, 1500, 2000, 2500]]

    dx = 5000/meshsize

    os.chdir(path)

    # define bed
    bed = [0.00 for i in range(np.int(meshsize+1))]
    bed_file = open('bed.dep', 'w+')
    bed_file.write(" ".join(str(x) for x in bed))
    bed_file.close()
    
    # set-up input parameter file with uncertain manning friction coefficient
    f = open('params.txt', 'w+')

    f.write("swave = 0")
    f.write("\n")
    f.write("nonh = 1")
    f.write("\n")
    f.write("front = nonh_1d")
    f.write("\n")
    f.write("left = wall")
    f.write("\n")
    f.write("right = wall")
    f.write("\n")
    f.write("back = wall")
    f.write("\n")
    f.write("ARC = 0")
    f.write("\n")
    f.write("order = 1")
    f.write("\n")
    f.write("nonhq3d      = 1")
    f.write("\n")
    f.write("bedfriction  = manning")
    f.write("\n")
    f.write("bedfriccoef  = ")
    f.write(str(sample))
    f.write("\n")
    f.write("dy = 1")
    f.write("\n")
    f.write("dx = ")
    f.write(str(dx))
    f.write("\n")
    f.write("nx = ")
    f.write(str(np.int(meshsize)))
    f.write("\n")
    f.write("ny = 0")
    f.write("\n")
    f.write("depfile = bed.dep")
    f.write("\n")
    f.write("thetamin = 0")
    f.write("\n")
    f.write("thetamax = 360")
    f.write("\n")
    f.write("dtheta = 360")
    f.write("\n")
    f.write("zs0  = .01")
    f.write("\n")
    f.write("eps   = .001")
    f.write("\n")
    f.write("tstop = 3600")
    f.write("\n")
    f.write("CFL  = 0.25")
    f.write("\n")
    f.write("instat  = ts_nonh")
    f.write("\n")
    f.write("outputformat = netcdf")
    f.write("\n")
    f.write("tintg   = 3600")
    f.write("\n")
    f.write("tstart   = 0")
    f.write("\n")
    f.write("nglobalvar   = 1")
    f.write("\n")
    f.write("zs")

    f.close()

    # # copy relevant boundary conditions files into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Non-breaking_wave/boun_U.bcf", os.path.join(path,'boun_U.bcf'))

    # run model
    subprocess.check_call("/rds/general/user/mc4117/home/trunk/src/xbeach/xbeach > /dev/null", shell=True)

    dataset = Dataset('xboutput.nc')

    # post-process data and extract outputs of interest
    x_array = dataset.variables['globalx'][0].data
    dataarray = dataset.variables['zs'][len(dataset.variables['zs'])-1].data[0]

    h = interpolate.interp1d(x_array, dataarray)

    return [h(i) for i in [1000, 1500, 2000, 2500]]

def low_fidelity(sample, meshsize, L0, path):
    # low fidelity model uses SFINCS

    if not os.path.exists(path):
        os.makedirs(path)

    # if less than minimum level considered in algorithm, return 0
    if meshsize <= L0:
        return [0 for i in [1000, 1500, 2000, 2500]]

    dx = 5000/meshsize

    os.chdir(path)

    # define bed
    bed = [0.00 for i in range(np.int(meshsize+1))]
    bed_file = open('sfincs.dep', 'w+')
    bed_file.write(" ".join(str(x) for x in bed))
    bed_file.close()
    
    # define mask
    msk = [2]
    for i in range(1, np.int(meshsize)):
        msk.append(1)
        
    msk_file = open('sfincs.msk', 'w+')
    msk_file.write(" ".join(str(i) for i in msk))
    msk_file.close()    

    # set-up input parameter file with uncertain Manning friction coefficient
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
    f.write("tref = 20171019 000000")
    f.write("\n") 
    f.write("tstart         = 20171019 000000")
    f.write("\n") 
    f.write("tstop          = 20171019 010000")
    f.write("\n") 
    f.write("dtout = 3600")
    f.write("\n") 
    f.write("dthisout = 3600")
    f.write("\n") 
    f.write("dtwnd          = 1800")
    f.write("\n") 
    f.write("alpha          = 0.25")
    f.write("\n")
    f.write("manning        = ")
    f.write(str(sample))
    f.write("\n")
    f.write("zsini          = 0")
    f.write("\n") 
    f.write("qinf           = 0")
    f.write("\n") 
    f.write("depfile        = sfincs.dep")
    f.write("\n") 
    f.write("mskfile        = sfincs.msk")
    f.write("\n") 
    f.write("bndfile        = sfincs.bnd")
    f.write("\n") 
    f.write("bzsfile        = sfincs.bzs")
    f.write("\n") 
    f.write("inputformat    = asc")
    f.write("\n") 
    f.write("outputformat   = net")
    f.write("\n") 
    f.write("bndtype        = 3")
    f.write("\n") 
    f.write("theta          = 0.9")
    f.write("\n") 
    f.write("advection      = 1")

    f.close()

    cwd = os.getcwd()
    
    # copy relevant boundary conditions files into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Non-breaking_wave/sfincs.bnd", os.path.join(cwd,'sfincs.bnd'))

    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Non-breaking_wave/sfincs.bzs", os.path.join(cwd,'sfincs.bzs'))

    # run model
    subprocess.check_call("singularity run /rds/general/user/mc4117/home/SFINCS_MLMF/sfincs-cpu_latest.sif > /dev/null", shell=True)

    dataset = Dataset('sfincs_map.nc')

    # post-process data and extract outputs of interest
    x_array = dataset['x'][:].data
    zs_data =  dataset['zs'][len(dataset['zs'])-1].data

    dataarray = np.where(zs_data==-9.999900e+04, 0.01, zs_data) 

    xarray_mod = [i[0] for i in x_array]
    dataarray_mod = [i[0] for i in dataarray]    

    h = interpolate.interp1d(xarray_mod, dataarray_mod)   

    return [h(i) for i in [1000, 1500, 2000, 2500]]
