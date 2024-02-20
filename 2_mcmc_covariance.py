import lib.Utils as u
import os
import json

with open('map_params.json','r') as filename:
	f = filename.read()
map_pars = json.loads(f)

with open('mcmc_params.json','r') as filename:
	f = filename.read()
mcmc_pars = json.loads(f)

if mcmc_pars['mcmc_pix_dense'] == False:
	if mcmc_pars['mcmc_IQU_dense'] == False:
		if mcmc_pars['mcmc_nu_dense'] == False:
			cov_path = 'mcmc_diag/'
		else:
			cov_path = 'mcmc_NU_dense/'
	else:
                if mcmc_pars['mcmc_nu_dense'] == False:
                        cov_path = 'mcmc_IQU_dense/'
                else:
                        cov_path = 'mcmc_IQU-NU_dense/'
else:
        if mcmc_pars['mcmc_IQU_dense'] == False:
                if mcmc_pars['mcmc_nu_dense'] == False:
                        cov_path = 'mcmc_PIX_dense/'
                else:
                        cov_path = 'mcmc_PIX-NU_dense/'
        else:
                if mcmc_pars['mcmc_nu_dense'] == False:
                        cov_path = 'mcmc_PIX-IQU_dense/'
                else:
                        cov_path = 'mcmc_PIX-IQU-NU_dense/'


print('\nBuilding inverted matrices for component separation...')
u.build_mcmc_covs(nus       = map_pars['nus'],\
                  npix      = int(mcmc_pars['npix']),\
		  path_in   = map_pars['print_path'], path_out  = map_pars['print_path'] + cov_path,\
		  pix_dense = mcmc_pars['mcmc_pix_dense'], IQU_dense = mcmc_pars['mcmc_IQU_dense'], nu_dense = mcmc_pars['mcmc_nu_dense'],\
		  print_ = True)

print('Converting files in unformatted...')
os.system('gfortran -c lib/healpix_types.f90 lib/F90_inv_cov_writer.f90')
os.system('gfortran -o lib/F90_inv_cov_writer healpix_types.o F90_inv_cov_writer.o')
os.system('./lib/F90_inv_cov_writer')
os.system('rm F90_inv_cov_writer.o healpix_types*')
os.system('rm '+map_pars['print_path']+cov_path+'*.txt')
os.system('cp mcmc_params.json '+map_pars['print_path'] + cov_path)
for n in range(map_pars['noise_reals']):
	os.system('mkdir '+map_pars['print_path'] + cov_path+'REAL_'+str(n+1))

