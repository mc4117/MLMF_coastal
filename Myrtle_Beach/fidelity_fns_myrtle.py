"""
File defines high and low fidelity models

The models solve the Myrtle Beach test case for the uncertain parameter of the maximum tide height.

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
        return [0 for i in range(9)]

    os.chdir(path)

    cwd = os.getcwd()
    nmax = np.int(np.ceil(meshsize*38.75))
    dy = np.ceil(1240/nmax)

    # set-up parameter file and specify the locations of interest
    f = open('params.txt', 'w+')
    f.write("wavemodel   = surfbeat")
    f.write("\n")
    f.write("wbctype     = jonstable")
    f.write("\n")
    f.write("bcfile     = jonswap.txt")
    f.write("\n")
    f.write("depfile     = xbeach.dep")
    f.write("\n")
    f.write("posdwn      = 0")
    f.write("\n")
    f.write("gridform    = delft3d")
    f.write("\n")
    f.write("xyfile      = xbeach.grd")
    f.write("\n")
    f.write("thetamin    = 0")
    f.write("\n")
    f.write("thetamax    = 180")
    f.write("\n")
    f.write("dtheta      = 10")
    f.write("\n")
    f.write("tstop       = 10800")
    f.write("\n")
    f.write("zs0 = -1.0")
    f.write("\n")
    f.write("CFL = 0.9")
    f.write("\n")
    f.write("bedfriction = manning")
    f.write("\n")
    f.write("bedfriccoef = 0.02")
    f.write("\n")
    f.write("sedtrans    = 0")
    f.write("\n")
    f.write("morphology  = 0")
    f.write("\n")
    f.write("zs0file     = tide.txt")
    f.write("\n")
    f.write("tideloc     = 2")
    f.write("\n")
    f.write("tintm       = 10800")
    f.write("\n")
    f.write("tintp       = 10")
    f.write("\n")
    f.write("tintg       = 10800")
    f.write("\n")
    f.write("nglobalvar = 0")
    f.write("\n")
    f.write("npointvar   = 1")
    f.write("\n")
    f.write("zs")
    f.write("\n")
    f.write("npoints     = 8")
    f.write("\n")
    f.write("707279 3740010")
    f.write("\n")
    f.write("707596 3740390")
    f.write("\n")
    f.write("708083 3740530")
    f.write("\n")
    f.write("708368 3740730")
    f.write("\n")
    f.write("707497 3740650")
    f.write("\n")
    f.write("707113 3739790")
    f.write("\n")
    f.write("708586 3740480")
    f.write("\n")
    f.write("707455 3739690")
    f.close()

    # set-up tide file with uncertain maximum tide height
    g = open('tide.txt', 'w+')   
    g.write("0.0000000e+00   0.0000000e+00   0.0000000e+00")
    g.write("\n")
    g.write("3.6000000e+03   ")
    g.write(str(sample))
    g.write("   0.0000000e+00")
    g.write("\n")
    g.write("7.2000000e+03   ")
    g.write(str(sample)) 
    g.write("   0.0000000e+00")
    g.write("\n")
    g.write("1.0800000e+04   0.0000000e+00   0.0000000e+00")
    g.close()
    
    # copy grids and beds for the resolution being considered into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/xbeach_grids/XBeach_dx"+ str(np.int(dy)) + "m/xbeach.dep", os.path.join(cwd,'xbeach.dep'))
    
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/xbeach_grids/XBeach_dx"+ str(np.int(dy)) + "m/xbeach.grd", os.path.join(cwd,'xbeach.grd'))  
    
    # copy wave spectrum definition file into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/xbeach_grids/XBeach_dx"+ str(np.int(dy)) + "m/jonswap.txt", os.path.join(cwd,'jonswap.txt'))  

    # run model
    subprocess.check_call("/rds/general/user/mc4117/home/trunk/src/xbeach/xbeach > /dev/null", shell=True)

    dataset = Dataset('xboutput.nc')

    # post-process data and extract outputs of interest
    zs_data = dataset['point_zs']    

    return [zs_data[:, i].data.max() for i in range(zs_data.shape[1])]


def low_fidelity(sample, meshsize, L0, path):
    # low fidelity model uses SFINCS

    if not os.path.exists(path):
        os.makedirs(path)

    # if less than minimum level considered in algorithm, return 0
    if meshsize <= L0:
        return [0 for i in range(9)]

    os.chdir(path)


    cwd = os.getcwd()
    nmax = np.int(np.ceil(meshsize*38.75))
    dy = np.ceil(1240/nmax)
    
    # set-up parameter file
    f = open('sfincs.inp', 'w+')
    f.write("mmax = 231")    
    f.write("\n")
    f.write("nmax = ")   
    f.write(str(nmax))
    f.write("\n")
    f.write("dx = 10")
    f.write("\n")
    f.write("dy = ")
    f.write(str(dy))
    f.write("\n")
    f.write("x0  = 707320.8381")
    f.write("\n") 
    f.write("y0  = 3739123.3192")
    f.write("\n")
    f.write("rotation  = 34.3")
    f.write("\n") 
    f.write("latitude = 0")
    f.write("\n")     
    f.write("tref = 20210716 000000")
    f.write("\n") 
    f.write("tstart         = 20210716 000000")
    f.write("\n") 
    f.write("tstop          = 20210716 030000")
    f.write("\n")
    f.write("tspinup = 60")
    f.write("\n")     
    f.write("dtmapout = 10800")
    f.write("\n") 
    f.write("dthisout = 10")
    f.write("\n")
    f.write("dtmaxout = 10800")
    f.write("\n")
    f.write("dtwnd          = 1800")
    f.write("\n") 
    f.write("alpha          = 0.5")
    f.write("\n")
    f.write("theta          = 0.9")
    f.write("\n")
    f.write("huthresh          = 0.005")
    f.write("\n")    
    f.write("manning_land        = 0.02")
    f.write("\n")
    f.write("manning_sea        = 0.02")
    f.write("\n")
    f.write("rgh_lev_land    = 0")
    f.write("\n")    
    f.write("zsini          = -1.0")
    f.write("\n") 
    f.write("qinf           = 0")
    f.write("\n")
    f.write("rhoa           = 1.25")
    f.write("\n")
    f.write("rhow           = 1024")
    f.write("\n")
    f.write("dtmax          = 999")
    f.write("\n")
    f.write("maxlev         = 999")
    f.write("\n")
    f.write("bndtype        = 1")
    f.write("\n")
    f.write("advection      = 0")
    f.write("\n")
    f.write("baro           = 0")
    f.write("\n")
    f.write("pavbnd         = 0")
    f.write("\n")
    f.write("gapres         = 101200")
    f.write("\n")
    f.write("advlim         = 9999.9")
    f.write("\n")
    f.write("stopdepth      = 100")
    f.write("\n")
    f.write("depfile        = sfincs.dep")
    f.write("\n")     
    f.write("mskfile        = sfincs.msk")
    f.write("\n")
    f.write("indexfile        = sfincs.ind")
    f.write("\n")    
    f.write("bndfile        = sfincs.bnd")
    f.write("\n")    
    f.write("bzsfile        = sfincs.bzs")
    f.write("\n")
    f.write("inputformat    = bin")
    f.write("\n") 
    f.write("outputformat   = net")
    f.write("\n") 
    f.write("obsfile   = sfincs.obs")
    f.write("\n") 

    f.close()
   
    # set-up tide file with uncertain maximum tide height
    g = open('sfincs.bzs', 'w+')
    g.write("0  0.000")
    g.write("\n")
    g.write("3600  ")
    g.write(str(sample))
    g.write("\n")
    g.write("7200  ")
    g.write(str(sample)) 
    g.write("\n")
    g.write("10800  0.000")
    g.close()

    # copy bed for resolution being considered into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/sfincs_grids/SFINCS_dx"+ str(np.int(dy)) + "m/sfincs.dep", os.path.join(cwd,'sfincs.dep'))
    
    # copy other set-up files into folder where model is run
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/sfincs_grids/SFINCS_dx"+ str(np.int(dy)) + "m/sfincs.bnd", os.path.join(cwd,'sfincs.bnd'))   

    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/sfincs_grids/SFINCS_dx"+ str(np.int(dy)) + "m/sfincs.ind", os.path.join(cwd,'sfincs.ind'))
        
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/sfincs_grids/SFINCS_dx"+ str(np.int(dy)) + "m/sfincs.msk", os.path.join(cwd,'sfincs.msk'))

    # copy file specifying the locations of interest
    copyfile("/rds/general/user/mc4117/home/MLMF_coastal/Myrtle_Beach/sfincs_grids/SFINCS_dx"+ str(np.int(dy)) + "m/sfincs.obs", os.path.join(cwd,'sfincs.obs'))
     
    # run model
    subprocess.check_call("singularity run /rds/general/user/mc4117/home/SFINCS_MLMF/sfincs-cpu_latest.sif > /dev/null", shell=True)

    dataset = Dataset('sfincs_his.nc')
    
    # post-process data and extract outputs of interest
    zs_data = dataset['point_zs']

    return [zs_data[:, i].data.max() for i in range(zs_data.shape[1])]
    
    
