import lib.Instrument as I
import lib.Utils as utils
import lib.MCMC as MCMC
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import sys
import json

with open('map_params.json','r') as filename:
	f = filename.read()

map_params = json.loads(f)

with open('mcmc_params.json','r') as filename:
	f = filename.read()

mcmc_params = json.loads(f)


if mcmc_params['mcmc_pix_dense'] == False:
	if mcmc_params['mcmc_IQU_dense'] == False:
		if mcmc_params['mcmc_nu_dense'] == False:
			cov_path = 'mcmc_diag/'
		else:
			cov_path = 'mcmc_NU_dense/'
	else:
		if mcmc_params['mcmc_nu_dense'] == False:
			cov_path = 'mcmc_IQU_dense/'
		else:
			cov_path = 'mcmc_IQU-NU_dense/'
else:
	if mcmc_params['mcmc_IQU_dense'] == False:
		if mcmc_params['mcmc_nu_dense'] == False:
			cov_path = 'mcmc_PIX_dense/'
		else:
			cov_path = 'mcmc_PIX-NU_dense/'
	else:
		if mcmc_params['mcmc_nu_dense'] == False:
			cov_path = 'mcmc_PIX-IQU_dense/'
		else:
			cov_path = 'mcmc_PIX-IQU-NU_dense/'


inst = I.Instrument(map_params,build_cov = False)

noise_realization = str(sys.argv[1])
mcmc_realization  = str(sys.argv[2])
mcmc_params['noise_realization'] = noise_realization
mcmc_params['mcmc_realization']  = mcmc_realization

## Building mcmc covariance
mcmc_inv_cov = utils.build_mcmc_covs(nus = map_params['nus'],\
          	 	             npix = int(mcmc_params['npix']),\
                  	     	     path_in = map_params['print_path'], path_out = '',\
                  	      	     pix_dense = mcmc_params['mcmc_pix_dense'], IQU_dense =  mcmc_params['mcmc_IQU_dense'],nu_dense = mcmc_params['mcmc_nu_dense'],\
                               	     print_ = False, return_ = True)

inst.mcmc_inv_subcovs = {}
for i,f1 in enumerate(inst.nus):
	for j,f2 in enumerate(inst.nus):
		if str(f2)+'_'+str(f1) in inst.mcmc_inv_subcovs.keys():
			continue
		
		inst.mcmc_inv_subcovs[str(f1)+'_'+str(f2)] = mcmc_inv_cov[i*3*inst.npix:(i+1)*3*inst.npix,j*3*inst.npix:(j+1)*3*inst.npix].copy()


with open(map_params['print_path']+'input_parameters.json') as filename:
	f = filename.read()

data = json.loads(f)
data['maps'] = {}
for f in inst.nus:
	map_ = hp.read_map(map_params['print_path']+'REAL_'+noise_realization+'/signal_map_'+str(f)+'.fits',field=None)
	data['maps'][str(f)] = np.zeros((3,inst.npix))
	for i in range(3):
		data['maps'][str(f)][i] = map_[i][inst.patch_pixels].copy()



## CompSep global parameters
if not mcmc_params['mcmc_pix_dense']:
	PIXEL = int(mcmc_params['pixel'])
else:
	PIXEL = 0  # just for plotting results

params = map_params.copy(); params.update(mcmc_params)
mcmc = MCMC.MCMC(data,inst,params)

## Launch MCMC
mcmc.decorr_metropolis(pix=PIXEL, stokes = '', optimize=False)


## Save results
fig,ax = plt.subplots(mcmc.npars,2,figsize=(15,mcmc.npars*4))
for i,name in enumerate(mcmc.names):
	ax[i,0].plot(mcmc.samples[:,PIXEL,i])
	ax[i,0].set_title(name)
	ax[i,0].vlines(mcmc.burn_in, mcmc.samples[:,PIXEL,i].min(), mcmc.samples[:,PIXEL,i].max(), label= "burn-in", colors="r")
	ax[i,0].hlines(data[name][PIXEL],0, mcmc.it, label = "True value", colors = "c")
	ax[i,0].legend(loc="upper left")
	
	y, x, _ = ax[i,1].hist(mcmc.samples[mcmc.burn_in:,PIXEL,i], bins=min(50, int(np.sqrt(mcmc.it))))
	ax[i,1].vlines(data[name][PIXEL], 0, y.max(), label = "True value", colors = "c")
	ax[i,1].vlines(np.average(mcmc.samples[mcmc.burn_in:,PIXEL,i]), 0, y.max(), label="Average", colors="r")
	ax[i,1].set_title(name)
	ax[i,1].legend(loc="upper left")
fig.savefig(map_params['print_path']+cov_path+'REAL_'+noise_realization+\
            '/Param_chains_n'+mcmc_realization+'.png',dpi=300)


mcmc.plot_results(png_title = map_params['print_path']+cov_path+'REAL_'+noise_realization+\
                  '/Freq_spectra_n'+mcmc_realization)

stream = open(map_params['print_path']+cov_path+'REAL_'+noise_realization+\
              '/Chain_log_n'+mcmc_realization+'.txt','w')
stream.write('Iterations: '+str(mcmc.it)+'\n')
stream.write('Burn-in: '+str(mcmc.burn_in)+'\n')
stream.write('Acceptance rate: '+str(mcmc.accept_rate/mcmc.it)+'\n\n')
for i,name in enumerate(mcmc.AMPLITUDES.keys()):
	stream.write(name+'\n')
	stream.write('MEAN: ' +str(np.average(mcmc.samples[mcmc.burn_in:,PIXEL,i]))+'\n')
	stream.write('STDV: ' +str(np.sqrt(mcmc.variances[PIXEL,i]))+'\n')
	stream.write('TRUE: ' +str(data[name][PIXEL])+'\n')
	stream.write('\n')
	for n in range(mcmc.it):
		stream.write(str(mcmc.samples[n,PIXEL,i]))
		stream.write('\n')
	stream.write('\n')
for i,name in enumerate(mcmc.SPEC_PARAMS.keys()):
	stream.write(name+'\n')
	stream.write('MEAN: ' +str(np.average(mcmc.samples[mcmc.burn_in:,PIXEL,i+3*len(mcmc.components)]))+'\n')
	stream.write('STDV: ' +str(np.sqrt(mcmc.variances[PIXEL,i+3*len(mcmc.components)]))+'\n')
	stream.write('TRUE: ' +str(data[name][PIXEL])+'\n')
	stream.write('\n')
	for n in range(mcmc.it):
		stream.write(str(mcmc.samples[n,PIXEL,i+3*len(mcmc.components)]))
		stream.write('\n')
	stream.write('\n')

stream.close()

os.system('scp mcmc_params.json '+map_params['print_path']+cov_path)

