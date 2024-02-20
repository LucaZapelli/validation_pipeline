import lib.Utils as utils
import numpy as np
import healpy as hp
import os
import json

with open('map_params.json','r') as filename:
	f = filename.read()

map_pars = json.loads(f)

with open('mcmc_params.json','r') as filename:
	f = filename.read()

mcmc_pars = json.loads(f)

print('\nImporting covariance used in map making...')
nus = [float(x)*1e9 for x in map_pars['nus']]
npix = mcmc_pars['npix']
full_cov = np.zeros((3*npix*len(nus),3*npix*len(nus)))
subcov_names = []
for i,f1 in enumerate(nus):
        for j,f2 in enumerate(nus):
                if str(f2)+'_'+str(f1) in subcov_names:
                        continue              
                covname = str(f1)+'_'+str(f2)
                subcov_names.append(covname)
                filename = map_pars['print_path']+'TRUE_cov_'+covname+'.txt'
                stream = open(filename,'r')
                text = stream.readlines()
                stream.close()
                mat = [] 
                for line in text:
                        mat.append([float(el) for el in line[:-1].split('\t')])
                mat = np.array(mat)
                full_cov[3*npix*i:3*npix*(i+1),3*npix*j:3*npix*(j+1)] = mat.copy()


print('\nImporting sky maps...')
with open(map_pars['print_path']+'input_parameters.json','r') as filename:
    f = filename.read()

input_pars = json.loads(f)

component_maps = {}
for comp in input_pars['components']:
        component_maps[comp] = {'I': {}, 'Q': {}, 'U': {}}
        for s in ['I','Q','U']:
                component_maps[comp][s] = {}
                for f in nus:
                        component_maps[comp][s][str(f)] = np.array(input_pars[comp+'_'+s+'_'+str(f)])


mask = hp.read_map(map_pars['print_path']+'mask.fits',field=None)
patch_pixels = []
for p,pix in enumerate(mask[0]):
        if pix == 1:
                patch_pixels.append(p)

MAPS = {}
npixs = 12 * map_pars['nside']**2
for f in nus:
        MAPS[str(f)] = np.zeros((3,npixs))
        for s,stokes in enumerate(['I','Q','U']):
                for comp in input_pars['components']:
                        MAPS[str(f)][s][patch_pixels] += component_maps[comp][stokes][str(f)].copy()


## -- Noise
numbering = ['st','nd','rd']

for n in range(map_pars['noise_reals']):
        if n < 3:
                print('Injecting '+str(n+1)+numbering[n]+' noise realization...')
        else:
                print('Injecting '+str(n+1)+'th noise realization...')

        n_ = n + map_pars['noise_reals_offset']
        if map_pars['print_path'] != '':
                os.system('mkdir '+map_pars['print_path']+'REAL_'+str(n_+1))

        full_noise_terms = utils.generate_corr_vars(full_cov,seed=map_pars['noise_seed']+n_).reshape(((len(nus),3,npix)))
        for i1,f1 in enumerate(nus):
                noise_map = np.zeros((3,npixs))

                for s in range(3):
                        for p,pix in enumerate(patch_pixels):
                                noise_map[s,pix] = full_noise_terms[i1,s,p]

                Signal = MAPS[str(f1)] + noise_map
                Signal *= mask

                if map_pars['print_path'] != '':
                        hp.write_map(map_pars['print_path']+"REAL_"+str(n_+1)+"/signal_map_" + str(f1) + ".fits",Signal,column_units='$\mu K_{CMB}$', overwrite=True, dtype=np.dtype('float64'))



