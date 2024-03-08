import lib.Instrument as I
import lib.Utils as utils
import numpy as np
import healpy as hp
import scipy.constants as sp
import scipy.linalg as la
import astropy
import camb
import pysm3
import pysm3.units as u
from pysm3.utils import normalize_weights
import os
import sys
import json

with open('map_params.json','r') as filename:
	f = filename.read()

glob_pars = json.loads(f) 

if glob_pars['bps'] == 'delta':
	glob_pars['bwidths'] = np.array(glob_pars['bwiths'])*0

os.system('cp sbatch.sh map_params.json '+glob_pars['print_path'])

################
## INSTRUMENT ##
################
print('Initializing instrument...')

inst = I.Instrument(glob_pars)

## -- Banpasses
if glob_pars['print_path'] != '':
	step = 0.01 # 10 MHz
	for i,f in enumerate(inst.nus/1e9):
		if os.path.exists(glob_pars['print_path']+'bp_'+str(f)+'.dat'):
			os.system('rm '+glob_pars['print_path']+'bp_'+str(f)+'.dat')
		stream=open(glob_pars['print_path']+'bp_'+str(f)+'.dat','a')
		start = int(inst.edges[i][0]/1e9)
		end = int(inst.edges[i][1]/1e9)
		for n in range(int((end - start)/step)+1):
			x = round(start + n*step,2)
			stream.write(str(x) + '\t1\n')

## -- Smoothing tools
pixwin = hp.pixwin(inst.nside, pol=True, lmax=inst.nside*4)
if glob_pars['print_path'] != '':
	N_digits = int(np.log10(inst.nside)) + 1
	#os.system('scp /path/to/pixel_window_n'+\
        #          '0'*(4-N_digits) + str(inst.nside) + '.fits '+glob_pars['print_path'])

	for i,f in enumerate(inst.nus/1e9):
		beam = hp.gauss_beam(fwhm=inst.fwhm[i] * np.pi/(180*60),lmax=inst.nside*4,pol=True)[:,0]
		hp.write_cl(glob_pars['print_path']+'beam_'+str(f)+'.fits',beam,overwrite=True)


## -- Mask

mask = np.array([inst.mask,inst.mask,inst.mask])
if glob_pars['print_path'] != '':
	hp.write_map(glob_pars['print_path']+"mask.fits",mask,overwrite=True)



################
## MAP MAKING ##
################

data = {'components': []}
MAPS = {}

## -- CMB
print('Generating cmb...')
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.7, ombh2=0.0224, omch2=0.12, mnu=0.06, omk=0.001, tau=0.054, Alens=1.) ## from Planck VI
pars.InitPower.set_params(As=2e-9, ns=0.965, r = glob_pars['r'])
if glob_pars['r'] == 0:
	pars.WantTensors = False
else:
	pars.WantTensors = True

pars.set_for_lmax(2500, lens_potential_accuracy=1)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
Dl = powers['total']
el = np.arange(2551)
Cl = np.zeros((4,2551))
for i in range(4):
	Cl[i,2:] = Dl[2:,i]*(2*np.pi)/(el[2:]*(el[2:]+1))

if glob_pars['cmb_seed'] == 0:
    seed = int(np.random.uniform(10**6))
else:
    seed = glob_pars['cmb_seed']

np.random.seed(seed)
cmb = hp.synfast(Cl, nside=inst.nside, new=True)

for i,f in enumerate(inst.nus):
	if glob_pars['smoothing']:
		cmb_ = pysm3.apply_smoothing_and_coord_transform(cmb,\
		fwhm=astropy.units.Quantity(inst.fwhm[i] * np.pi/(180*60) * u.rad), lmax=inst.nside*4)

		if inst.nside >= 32:
			cmb_[0] = hp.smoothing(cmb_[0],\
	                          beam_window=pixwin[0], use_pixel_weights = True,iter=0,pol=False)
			cmb_[1] = hp.smoothing(cmb_[1],\
	                          beam_window=pixwin[1], use_pixel_weights = True,iter=0,pol=False)
			cmb_[2] = hp.smoothing(cmb_[2],\
	                          beam_window=pixwin[1], use_pixel_weights = True,iter=0,pol=False)
		else:
			cmb_[0] = hp.smoothing(cmb_[0],\
	                          beam_window=pixwin[0], use_weights = True,iter=0,pol=False)
			cmb_[1] = hp.smoothing(cmb_[1],\
	                          beam_window=pixwin[1], use_weights = True,iter=0,pol=False)
			cmb_[2] = hp.smoothing(cmb_[2],\
	                          beam_window=pixwin[1], use_weights = True,iter=0,pol=False)
	
		MAPS[str(f)] = cmb_.copy()
	else:
		MAPS[str(f)] = cmb.copy()

# Collecting data
data['components'].append('cmb')
data['a_cmb_I'] = list(cmb[0][inst.patch_pixels].copy())
data['a_cmb_Q'] = list(cmb[1][inst.patch_pixels].copy())
data['a_cmb_U'] = list(cmb[2][inst.patch_pixels].copy())
for f in inst.nus:
	data['cmb_I_'+str(f)] = list(cmb[0][inst.patch_pixels].copy())
	data['cmb_Q_'+str(f)] = list(cmb[1][inst.patch_pixels].copy())
	data['cmb_U_'+str(f)] = list(cmb[2][inst.patch_pixels].copy())


## -- Foregrounds
print('Generating foregrounds...')
fgs_list = glob_pars['components'].copy()
fgs_list.remove('cmb')

for c in fgs_list:

	if c[0] == 's':  # Synchrotron
		fg_name = 'sync'
	if c[0] == 'd':  # Dust
		fg_name = 'dust'
	

	obj = pysm3.Sky(nside=inst.nside, preset_strings=[c])
	obj.output_unit = astropy.units.core.PrefixUnit('uK_CMB')

	for i,f in enumerate(inst.nus):
		fg = obj.get_emission(inst.edges[i] * u.Hz).value
		if glob_pars['smoothing']:
			fg = pysm3.apply_smoothing_and_coord_transform(fg,\
	                     fwhm=astropy.units.Quantity(inst.fwhm[i] * (np.pi/(180*60)) * u.rad), lmax=inst.nside*4)		
			if inst.nside >= 32:
				fg[0] = hp.smoothing(fg[0],\
					beam_window=pixwin[0], use_pixel_weights=True, iter=0, pol=False)
				fg[1] = hp.smoothing(fg[1],\
					beam_window=pixwin[1], use_pixel_weights=True, iter=0, pol=False)
				fg[2] = hp.smoothing(fg[2],\
					beam_window=pixwin[1], use_pixel_weights=True, iter=0, pol=False)
			else:
				fg[0] = hp.smoothing(fg[0],\
					beam_window=pixwin[0], use_weights=True, iter=0, pol=False)
				fg[1] = hp.smoothing(fg[1],\
					beam_window=pixwin[1], use_weights=True, iter=0, pol=False)
				fg[2] = hp.smoothing(fg[2],\
					beam_window=pixwin[1], use_weights=True, iter=0, pol=False)
		
		
		MAPS[str(f)] += fg.copy()
		
		## conversion for JSON serialization
		FG1 = np.array(fg[0][inst.patch_pixels].copy(),dtype='float64')
		FG2 = np.array(fg[1][inst.patch_pixels].copy(),dtype='float64')
		FG3 = np.array(fg[2][inst.patch_pixels].copy(),dtype='float64')
		data[fg_name+'_I_'+str(f)] = list(FG1)
		data[fg_name+'_Q_'+str(f)] = list(FG2)
		data[fg_name+'_U_'+str(f)] = list(FG3)
	
	# Collecting data
	if c[0] == 's':  # Synchrotron
		data['components'].append('synchrotron')
		data['local_ref_I_sync'] = inst.nus[0]
		data['local_ref_P_sync'] = inst.nus[0]
		data['ref_curv_sync'] = inst.nus[0]
        
		data['beta_sync'], data['curv_sync'] = np.zeros(inst.npix), np.zeros(inst.npix)
		if type(obj.components[0].pl_index.value) == np.float64:
			data['beta_sync'] += obj.components[0].pl_index.value.copy()
		else:
			data['beta_sync'] += obj.components[0].pl_index.value[inst.patch_pixels].copy()
		if hasattr(obj.components[0], 'spectral_curvature'):
			if type(obj.components[0].spectral_curvature) == float:
				data['curv_sync'] += obj.components[0].spectral_curvature.copy()
			else:
				data['curv_sync'] += obj.components[0].spectral_curvature[inst.patch_pixels].copy()
			data['ref_curv_sync'] = obj.components[0].freq_curve.copy() * 1e9   ## Keeping pysm reference
		data['beta_sync'] = list(data['beta_sync'])
		data['curv_sync'] = list(data['curv_sync'])

		
	if c[0] == 'd':  # Dust
		data['components'].append('dust')
		data['local_ref_I_dust'] = inst.nus[-1]
		data['local_ref_P_dust'] = inst.nus[-1]
            
		data['beta_dust'], data['temp_dust'] = np.zeros(inst.npix), np.zeros(inst.npix) 
		if type(obj.components[0].mbb_index.value) == np.float64:
			data['beta_dust'] += obj.components[0].mbb_index.value.copy()
		else:
			data['beta_dust'] += obj.components[0].mbb_index.value[inst.patch_pixels].copy()
		if type(obj.components[0].mbb_temperature.value) == np.float64:
			data['temp_dust'] += obj.components[0].mbb_temperature.value.copy()
		else:
			data['temp_dust'] += obj.components[0].mbb_temperature.value[inst.patch_pixels].copy()
		data['beta_dust'] = list(data['beta_dust'])
		data['temp_dust'] = list(data['temp_dust'])
                
	## (smoothed) amplitude converted at local reference frequency
	em_ref = obj.get_emission(data['local_ref_I_'+fg_name] * u.Hz)[0].value
            
	if glob_pars['smoothing']:
		em_ref = pysm3.apply_smoothing_and_coord_transform(em_ref,\
                         fwhm=astropy.units.Quantity(inst.fwhm * u.rad), lmax=inst.nside*4).value
		if inst.nside >= 32:
			em_ref = hp.smoothing(em_ref,\
                                 beam_window=pixwin[0], use_pixel_weights=True, iter=0, pol=False)
		else:
			em_ref = hp.smoothing(em_ref,\
                                 beam_window=pixwin[0], use_weights=True, iter=0, pol=False)
		    
	data['a_'+fg_name+'_I'] = em_ref[inst.patch_pixels].copy() * (1. * u.uK_CMB).to_value(u.uK_RJ,\
                                  equivalencies=u.cmb_equivalencies(data['local_ref_I_'+fg_name] * u.Hz))  ## uk_RJ
	data['a_'+fg_name+'_I'] = list(data['a_'+fg_name+'_I'])
             
    
	em_ref = obj.get_emission(data['local_ref_P_'+fg_name] * u.Hz).value
                
	if glob_pars['smoothing']:
		em_ref = pysm3.apply_smoothing_and_coord_transform(em_ref,\
                         fwhm=astropy.units.Quantity(inst.fwhm * u.rad), lmax=inst.nside*4).value
		if inst.nside >= 32:
			em_ref = hp.smoothing(em_ref,\
                                 beam_window=pixwin[1], use_pixel_weights=True, iter=0, pol=False)
		else:
			em_ref = hp.smoothing(em_ref,\
                                 beam_window=pixwin[1], use_weights=True, iter=0, pol=False)       
        
	data['a_'+fg_name+'_Q'] = em_ref[1][inst.patch_pixels].copy() * (1. * u.uK_CMB).to_value(u.uK_RJ,\
                                  equivalencies=u.cmb_equivalencies(data['local_ref_P_'+fg_name] * u.Hz))
	data['a_'+fg_name+'_U'] = em_ref[2][inst.patch_pixels].copy() * (1. * u.uK_CMB).to_value(u.uK_RJ,\
                                  equivalencies=u.cmb_equivalencies(data['local_ref_P_'+fg_name] * u.Hz))          
      
	data['a_'+fg_name+'_Q'] = list(data['a_'+fg_name+'_Q'])
	data['a_'+fg_name+'_U'] = list(data['a_'+fg_name+'_U'])



with open(glob_pars['print_path']+'input_parameters.json','w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)



## -- Noise
numbering = ['st','nd','rd']

for n in range(glob_pars['noise_reals']):
	if n < 3:
		print('Injecting '+str(n+1)+numbering[n]+' noise realization...')
	else:
		print('Injecting '+str(n+1)+'th noise realization...')

	if glob_pars['print_path'] != '':
		os.system('mkdir '+glob_pars['print_path']+'REAL_'+str(n+1))
		
	full_noise_terms = utils.generate_corr_vars(inst.inst_covariance,seed=glob_pars['noise_seed']+n).reshape(((inst.nbands,3,inst.npix)))
	for i1,f1 in enumerate(inst.nus):
		noise_map = np.zeros((3,inst.npixs))
	
		for s in range(3):
			for p,pix in enumerate(inst.patch_pixels):
				noise_map[s,pix] = full_noise_terms[i1,s,p]
	
		Signal = MAPS[str(f1)] + noise_map
		Signal *= mask	

		if glob_pars['print_path'] != '':
			hp.write_map(glob_pars['print_path']+"REAL_"+str(n+1)+"/signal_map_" + str(f1) + ".fits",Signal,column_units='$\mu K_{CMB}$', overwrite=True, dtype=np.dtype('float64'))



if glob_pars['print_path'] != '':
	for i1,f1 in enumerate(inst.nus):
		rms_map = np.zeros((3,inst.npixs))
		for i in range(3):
			rms_map[i][inst.patch_pixels] = inst.noise_rms[i1,i].copy()
			hp.write_map(glob_pars['print_path']+"rms_map_" + str(f1) + ".fits",rms_map,column_units='$\mu K$', overwrite=True, dtype=np.dtype('float64'))
	
print('\nNPIX: '+str(inst.npix))






