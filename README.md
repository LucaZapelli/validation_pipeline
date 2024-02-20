###
### E2E Validation Pipeline
###

### 1. Map Making
 > Insert parameters for the map making into 'map_params.json'
 > Run '1_mapmaking.py'. This will produce:
    > "True" covariance block matrices
    > beam Cls
    > bandpass profiles
    > mask
    > pixel windown function Cls
    > rms maps
    > 'REAL_#' folders with signal maps


### 2. Covariance definition for Component Separation
 > Open lib/F90_inv_cov_writer.f90 and change the paths to where you want your data to be stored
   (this exe prints out commander-formatted block matrices)
 > Insert parameters for the Component Separation into 'mcmc_params.json'
 > Run '2_mcmc_covariance.py'


### 3. Component Separation
 > Run '3_my_component_separation.py'. This will produce a folder associated to the choosen 
   covariance configuration and noise realization, containing:
    > Chain logs
    > Frequency spectra reconstruction (plots)
    > Parameters chains (plots)

### 4. Data Analysis
 > Run 'Chains_anaylsis.ipynb'


