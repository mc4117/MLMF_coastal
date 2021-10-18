# MLMF_coastal

This repository provides a Python wrapper to implement the Multilevel Multifidelity Monte Carlo method (MLMF).

Here this wrapper has been implemented around the high fidelity model XBeach and the low fidelity model SFINCS to assess uncertainty in coastal problems.

The functions to implement MLMF are stored in `mlmf_fns_multi.py', a copy of which is included in every test case folder. Note this wrapper is model independent and thus it can be easily implemented with different models simply by changing the `fidelity_fns_*.py' file.

This repository also contains python code to transform MLMF expectation outputs into cumulative distribution functions using the modified inverse transform sampling method. The code to do this for each test case is found in the python file `*_dist.py' in each folder.

Finally, the functions required for performing Monte Carlo to verify the MLMF method are stored in `mc_fns.py`

Software requirements
-------------------------

1. XBeach (https://oss.deltares.nl/web/xbeach/)
    * In this work, we used version 1.23.5526 XBeachX release
2. SFINCS (Available on request. Please contact Deltares if you are interested in using this code.)
3. Python 3.5 or later


Simulation scripts
------------------

* Carrier-Greenspan

   ```
   #!bash
    $ python carrier_mlmf.py
   ```
    produces MLMF results if prelim_run = True; opt_sample = True; opt_hf=False, opt_lf=False;
    
    produces MLMC results for XBeach with prelim_run = True; opt_sample = False; opt_hf=True, opt_lf=False;
    
    produces MLMC results for SFINCS with prelim_run = True; opt_sample = False; opt_hf=False, opt_lf=True.
    
   ```
   #!bash
    $ python carrier_dist.py
   ```
    post-processes the outputs of `carrier_mlmf.py' to produce the cumulative distribution function.

   ```
   #!bash
    $ python carrier_mc.py
   ```
    produces Monte Carlo (MC) results to verify the MLMF/MLMC results. N.B. this file must be run multiple times to achieve a sufficient number of samples.
    
   ```
   #!bash
    $ python carrier_analytic.py
   ```
    produces the analytical result to verify the MLMF/MLMC results for this test case.

* Myrtle Beach

   ```
   #!bash
    $ python myrtle_mlmf.py
   ```
    produces MLMF results if prelim_run = True; opt_sample = True; opt_hf=False, opt_lf=False;
    
    produces MLMC results for XBeach with prelim_run = True; opt_sample = False; opt_hf=True, opt_lf=False;
    
    produces MLMC results for SFINCS with prelim_run = True; opt_sample = False; opt_hf=False, opt_lf=True.
    
   ```
   #!bash
    $ python myrtle_dist.py
   ```
    post-processes the outputs of `myrtle_mlmf.py' to produce the cumulative distribution function.

* Non-breaking_wave

   ```
   #!bash
    $ python bates_mlmf.py
   ```
    produces MLMF results if prelim_run = True; opt_sample = True; opt_hf=False, opt_lf=False;
    
    produces MLMC results for XBeach with prelim_run = True; opt_sample = False; opt_hf=True, opt_lf=False;
    
    produces MLMC results for SFINCS with prelim_run = True; opt_sample = False; opt_hf=False, opt_lf=True.
    
   ```
   #!bash
    $ python bates_dist.py
   ```
    post-processes the outputs of `bates_mlmf.py' to produce the cumulative distribution function.

   ```
   #!bash
    $ python bates_mc.py
   ```
    produces Monte Carlo (MC) results to verify the MLMF/MLMC results. N.B. this file must be run multiple times to achieve a sufficient number of samples.
    
   ```
   #!bash
    $ python bates_analytic.py
   ```
    produces the analytical result to verify the MLMF/MLMC results for this test case.
