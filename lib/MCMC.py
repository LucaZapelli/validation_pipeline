import numpy as np
import healpy as hp
import scipy.constants as sp
import scipy.linalg as la
import random
import matplotlib.pyplot as plt
import astropy
import pysm3
import pysm3.units as u
import pysm3.utils as utils
from pysm3.utils import normalize_weights
import os
from multiprocessing import Pool
from  matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import time
import scipy.stats as stats
from scipy.interpolate import interp1d
import corner
import sympy


class MCMC:
    
	def __init__(self,data,inst,parameters):
        
		#########-----  Utility variables  -----#########
		
		self.nside = inst.nside
		self.npixs = inst.npixs
		self.npix = inst.npix
		self.mask = inst.mask
		self.fwhm = inst.fwhm
		self.pixwin = hp.pixwin(self.nside, pol=True, lmax=self.nside*4)
		self.patch_pixels = inst.patch_pixels
		self.nus = inst.nus
		self.nbands = inst.nbands
		self.edges = inst.edges
		self.noise_rms = inst.noise_rms
		self.inv_cov_matrices = inst.mcmc_inv_subcovs
		self.data = data.copy()
		self.noise_realization = parameters['noise_realization']
		self.mcmc_realization = parameters['mcmc_realization']
		self.IQU_corr = parameters['mcmc_IQU_dense']
		self.pix_corr = parameters['mcmc_pix_dense']
		self.nu_corr = parameters['mcmc_nu_dense']
		self.path = parameters['print_path'] + parameters['cov_path'] + 'precond_REAL_' + self.noise_realization +'/' 

		self.AMPLITUDES, self.SPEC_PARAMS = {}, {}
		self.components = data['components']
		self.npars = 3 * len(self.components)
		for c in self.components:
			if c == 'cmb':
				self.AMPLITUDES.update({'a_cmb_I': np.zeros(self.npix),
							'a_cmb_Q': np.zeros(self.npix),
							'a_cmb_U': np.zeros(self.npix)})
			if c == 'synchrotron':
				self.sync_ref = {}
				self.sync_ref['I'] = data['local_ref_I_sync']
				self.sync_ref['Q'] = data['local_ref_P_sync']
				self.sync_ref['U'] = data['local_ref_P_sync']
				self.curv_ref = data['ref_curv_sync']
				self.AMPLITUDES.update({'a_sync_I': np.zeros(self.npix),
							'a_sync_Q': np.zeros(self.npix),
							'a_sync_U': np.zeros(self.npix)})
				self.SPEC_PARAMS.update({'beta_sync': np.zeros(self.npix),
							 'curv_sync': np.zeros(self.npix)})
				self.npars += 2
			if c == 'dust':
				self.dust_ref = {}
				self.dust_ref['I'] = data['local_ref_I_dust']
				self.dust_ref['Q'] = data['local_ref_P_dust']
				self.dust_ref['U'] = data['local_ref_P_dust']
				self.AMPLITUDES.update({'a_dust_I': np.zeros(self.npix),
							'a_dust_Q': np.zeros(self.npix),
							'a_dust_U': np.zeros(self.npix)})
				self.SPEC_PARAMS.update({'beta_dust': np.zeros(self.npix),
							 'temp_dust': np.zeros(self.npix)})
				self.npars += 2
		
		self.PRIORS = {}
		self.STEPS = {}
		self.names = []
		for key in self.AMPLITUDES:
			self.PRIORS[key] = np.array([-float("inf"),float("inf")])
			self.STEPS[key] = np.zeros(self.npix)
			self.names.append(key)
		for key in self.SPEC_PARAMS:
			self.PRIORS[key] = np.array([-float("inf"),float("inf")])
			self.STEPS[key] = np.zeros(self.npix)
			self.names.append(key)
		
		
		#########-----  MCMC variables  -----#########

		if 'iterations' in parameters.keys():
			self.it = parameters['iterations']
		else:
			self.it = 0
		if 'burn-in' in parameters.keys():
			self.burn_in = parameters['burn-in']
		else:
			self.burn_in = 0
            	
		self.mcmc_step = np.sqrt(parameters['chain_variance'])
		self.use_chains_covariance = parameters['use_chains_covariance']
		self.to_sample = []
		
		## Set intial values and priors
		for par in parameters['to_sample']:
			if par[:2] == 'a_':
				for s in ['I','Q','U']:
					self.to_sample.append(par+'_'+s)
					self.AMPLITUDES[par+'_'+s] += np.array(data[par+'_'+s]) + 50 * self.mcmc_step * (-1)**np.random.randint(2)
					ext = max(np.abs(np.array(data[par+'_'+s])))
					self.PRIORS[par+'_'+s] = np.array([-3*ext,3*ext])
					self.STEPS[par+'_'+s] += self.mcmc_step
			else:
				self.to_sample.append(par)
				self.SPEC_PARAMS[par] += np.array(data[par]) + 50 * self.mcmc_step * (-1)**np.random.randint(2)
				ext = max(np.abs(np.array(data[par])))
				self.PRIORS[par] = np.array([-3*ext,3*ext])
				self.STEPS[par] += self.mcmc_step		        
		
		### Importing templates
		if 'templates' in parameters.keys():
			for key in parameters['templates']:
				if key in self.AMPLITUDES.keys():
					self.AMPLITUDES[key] = np.array(data[key]).copy()
				if key in self.SPEC_PARAMS.keys():
					self.SPEC_PARAMS[key] = np.array(data[key]).copy()
				self.PRIORS[key] = np.array([-float("inf"),float("inf")])
				self.STEPS[key] = np.zeros(self.npix)

	def prior_cut(self):
		for par in self.to_sample:
			for pix in range(self.npix):
				if par in self.AMPLITUDES.keys():
					if self.AMPLITUDES[par][pix] < self.PRIORS[par][0] or self.AMPLITUDES[par][pix] > self.PRIORS[par][1]:
						return -float("inf")
				if par in self.SPEC_PARAMS.keys():
					if self.SPEC_PARAMS[par][pix] < self.PRIORS[par][0] or self.SPEC_PARAMS[par][pix] > self.PRIORS[par][1]:
						return -float("inf")
		return 0


	def decorr_prior_cut(self,pix,stokes):
		for par in self.to_sample:
			if stokes != '' and pix == -1:
				for p in range(self.npix):
					if par in self.AMPLITUDES.keys() and par[-1] == stokes:
						if self.AMPLITUDES[par][p] < self.PRIORS[par][0] or self.AMPLITUDES[par][p] > self.PRIORS[par][1]:
							return -float("inf")
					if par in self.SPEC_PARAMS.keys():
						if self.SPEC_PARAMS[par][p] < self.PRIORS[par][0] or self.SPEC_PARAMS[par][p] > self.PRIORS[par][1]:
							return -float("inf")
			if stokes != '' and pix != -1:
				if par in self.AMPLITUDES.keys() and par[-1] == stokes:
					if self.AMPLITUDES[par][pix] < self.PRIORS[par][0] or self.AMPLITUDES[par][pix] > self.PRIORS[par][1]:
						return -float("inf")
				if par in self.SPEC_PARAMS.keys():
					if self.SPEC_PARAMS[par][pix] < self.PRIORS[par][0] or self.SPEC_PARAMS[par][pix] > self.PRIORS[par][1]:
						return -float("inf")
			if stokes == '' and pix != -1:
				if par in self.AMPLITUDES.keys():
					if self.AMPLITUDES[par][pix] < self.PRIORS[par][0] or self.AMPLITUDES[par][pix] > self.PRIORS[par][1]:
						return -float("inf")
				if par in self.SPEC_PARAMS.keys():
					if self.SPEC_PARAMS[par][pix] < self.PRIORS[par][0] or self.SPEC_PARAMS[par][pix] > self.PRIORS[par][1]:
						return -float("inf")
		return 0 



	def conversion_factor(self, f, input_units=u.uK_RJ, output_units = u.uK_CMB):
		return (1. * input_units).to_value(output_units, equivalencies=u.cmb_equivalencies(f * u.Hz))   
	
	
	def sync_em(self, f, pix, stokes):
		return self.AMPLITUDES['a_sync_'+stokes][pix] *\
		       (f/self.sync_ref[stokes])**(self.SPEC_PARAMS['beta_sync'][pix] +\
                       self.SPEC_PARAMS['curv_sync'][pix] * np.log(f/self.curv_ref))
	
	
	def dust_em(self, f, pix, stokes):
		modul = (np.exp((sp.h * self.dust_ref[stokes])/(sp.k * self.SPEC_PARAMS['temp_dust'][pix]))-1)\
                        /(np.exp((sp.h * f)/(sp.k * self.SPEC_PARAMS['temp_dust'][pix]))-1)
		return self.AMPLITUDES['a_dust_'+stokes][pix] *\
                       (f/self.dust_ref[stokes])**(self.SPEC_PARAMS['beta_dust'][pix] + 1) * modul
	
	
	def emission(self,comp,f,pix,stokes):
		ff = self.edges[list(self.nus).index(f)]
		if comp == 'cmb':
			em = self.AMPLITUDES['a_cmb_'+stokes][pix]
			in_units = u.uK_CMB
		elif comp == 'synchrotron':
			em = self.sync_em(ff,pix,stokes)
			in_units = u.uK_RJ
		elif comp == 'dust':
			em = self.dust_em(ff,pix,stokes)
			in_units = u.uK_RJ
		
		if type(ff) == np.ndarray:
			weights = np.ones(len(ff))
			weights_to_input = (weights * in_units).to_value(
                                           (u.Jy / u.sr), equivalencies=u.cmb_equivalencies(ff * u.Hz))
			weights_to_output = (weights * u.uK_CMB).to_value(
                                            (u.Jy / u.sr), equivalencies=u.cmb_equivalencies(ff * u.Hz))
			factor = np.trapz(weights_to_input, ff)/np.trapz(weights_to_output, ff)
			w = normalize_weights(ff,weights) * factor    
			norm = em * w
			return np.trapz(norm,x=ff) 
		else:
			return em * self.conversion_factor(ff,in_units)


	def chi_sq(self):
		chisq = 0.        
                        
		for i1,f1 in enumerate(self.nus):
			for stokes1 in range(3):
				for pix1 in range(self.npix):
				
					for pix2 in range(self.npix): ## sum over pixels
						if pix1 != pix2 and not self.pix_corr:
							continue
							
						for stokes2 in range(3): ## sum over IQU
							if stokes1 != stokes2 and not self.IQU_corr:
								continue
							
							for i2,f2 in enumerate(self.nus):  ## sum over frequencies
								if f2 != f1 and not self.nu_corr:
									continue
									
								inv_cov_name = str(f1)+'_'+str(f2)
								if inv_cov_name not in self.inv_cov_matrices.keys():
									inv_cov_name = str(f2)+'_'+str(f1)
									cov_element = self.inv_cov_matrices[inv_cov_name].T[stokes1*self.npix+pix1, stokes2*self.npix+pix2]
								else:
									cov_element = self.inv_cov_matrices[inv_cov_name][stokes1*self.npix+pix1, stokes2*self.npix+pix2]
								if cov_element != 0:
									## LHS & RHS Signals
									S_LHS, S_RHS = 0, 0
									for c in self.components:
										S_LHS += self.emission(c,f1,pix1,['I','Q','U'][stokes1])
										S_RHS += self.emission(c,f2,pix2,['I','Q','U'][stokes2])
									
									chisq += (self.data['maps'][str(f1)][stokes1][pix1] - S_LHS) * \
										 cov_element * (self.data['maps'][str(f2)][stokes2][pix2] - S_RHS)
								
		return chisq
 


	def decorr_chi_sq(self,pix,st):
		chisq = 0.        
                        
		for i1,f1 in enumerate(self.nus):
		
			if st != '' and pix == -1:
				stokes = ['I','Q','U'].index(st)
				for pix1 in range(self.npix):
					for pix2 in range(self.npix): ## sum on pixels
						if pix1 != pix2 and not self.pix_corr:
							continue                    
						
						for i2,f2 in enumerate(self.nus):  ## sum on frequencies
							if f2 != f1 and not self.nu_corr:
								continue
							
							inv_cov_name = str(f1)+'_'+str(f2)
							if inv_cov_name not in self.inv_cov_matrices.keys():
								inv_cov_name = str(f2)+'_'+str(f1)
								cov_element = self.inv_cov_matrices[inv_cov_name].T[stokes*self.npix+pix1, stokes*self.npix+pix2]
							else:
								cov_element = self.inv_cov_matrices[inv_cov_name][stokes*self.npix+pix1, stokes*self.npix+pix2]
							if cov_element != 0:
								## LHS & RHS Signals
								S_LHS, S_RHS = 0, 0
								for c in self.components:
									S_LHS += self.emission(c,f1,pix1,['I','Q','U'][stokes])
									S_RHS += self.emission(c,f2,pix2,['I','Q','U'][stokes])
									
								chisq += (self.data['maps'][str(f1)][stokes][pix1] - S_LHS) * \
                                              				 cov_element * (self.data['maps'][str(f2)][stokes][pix2] - S_RHS)                    
			
			if st != '' and pix != -1:
				stokes = ['I','Q','U'].index(st)
				for i2,f2 in enumerate(self.nus):  ## sum on frequencies
					if f2 != f1 and not self.nu_corr:
						continue
					
					inv_cov_name = str(f1)+'_'+str(f2)
					if inv_cov_name not in self.inv_cov_matrices.keys():
						inv_cov_name = str(f2)+'_'+str(f1)
						cov_element = self.inv_cov_matrices[inv_cov_name].T[stokes*self.npix+pix, stokes*self.npix+pix]
					else:
						cov_element = self.inv_cov_matrices[inv_cov_name][stokes*self.npix+pix, stokes*self.npix+pix]
					if cov_element != 0:
						## LHS & RHS Signals
						S_LHS, S_RHS = 0, 0
						for c in self.components:
							S_LHS += self.emission(c,f1,pix,['I','Q','U'][stokes])
							S_RHS += self.emission(c,f2,pix,['I','Q','U'][stokes])
							
						chisq += (self.data['maps'][str(f1)][stokes][pix] - S_LHS) * \
							 cov_element * (self.data['maps'][str(f2)][stokes][pix] - S_RHS)                  
			
			if pix != -1 and st == '':
				for stokes1 in range(3):
					for stokes2 in range(3): ## sum on IQU
						if stokes1 != stokes2 and not self.IQU_corr:
							continue
							
						for i2,f2 in enumerate(self.nus):  ## sum on frequencies
							if f2 != f1 and not self.nu_corr:
								continue
						
							inv_cov_name = str(f1)+'_'+str(f2)
							if inv_cov_name not in self.inv_cov_matrices.keys():
								inv_cov_name = str(f2)+'_'+str(f1)
								cov_element = self.inv_cov_matrices[inv_cov_name].T[stokes1*self.npix+pix, stokes2*self.npix+pix]
							else:
								cov_element = self.inv_cov_matrices[inv_cov_name][stokes1*self.npix+pix, stokes2*self.npix+pix]
							if cov_element != 0:
								## LHS & RHS Signals
								S_LHS, S_RHS = 0, 0
								for c in self.components:
									S_LHS += self.emission(c,f1,pix,['I','Q','U'][stokes1])
									S_RHS += self.emission(c,f2,pix,['I','Q','U'][stokes2])
							
							chisq += (self.data['maps'][str(f1)][stokes1][pix] - S_LHS) * \
							         cov_element * (self.data['maps'][str(f2)][stokes2][pix] - S_RHS)
		return chisq
	
	
	
	def chain(self,i,optimize=False):
	
		if not self.use_chains_covariance:  ## preconditioning run
			for k,key in enumerate(self.AMPLITUDES.keys()):
				self.old_AMPLITUDES[key] = self.AMPLITUDES[key].copy()
				self.AMPLITUDES[key] += np.random.randn(self.npix) * self.STEPS[key]
		
			for k,key in enumerate(self.SPEC_PARAMS.keys()):
				self.old_SPEC_PARAMS[key] = self.SPEC_PARAMS[key].copy()
				self.SPEC_PARAMS[key] += np.random.randn(self.npix) * self.STEPS[key]  
		else:
			samps = np.zeros((self.npars,self.npix))
			for p in range(self.npix):
				samps[:,p] = self.FULL_SQRTCOV @ np.random.randn(self.npars)

			for k,key in enumerate(self.AMPLITUDES.keys()):
				self.old_AMPLITUDES[key] = self.AMPLITUDES[key].copy()
				self.AMPLITUDES[key] += samps[k]

			for k,key in enumerate(self.SPEC_PARAMS.keys()):
				self.old_SPEC_PARAMS[key] = self.SPEC_PARAMS[key].copy()
				self.SPEC_PARAMS[key] += samps[3 * len(self.components) + k]

		
		
		#########-----  Check Proposal  -----#########
		
		new_prob = -0.5 * self.chi_sq() + self.prior_cut()
		
		if not optimize: 
			A = min(1., np.exp(new_prob - self.old_prob))
			if A >= np.random.uniform(0,1):  
				self.old_prob = new_prob      
				self.accept_rate += 1
			else:
				for p in range(self.npix):
					for key in self.AMPLITUDES.keys():
						self.AMPLITUDES[key][p] = self.old_AMPLITUDES[key][p]
					for key in self.SPEC_PARAMS.keys():
						self.SPEC_PARAMS[key][p] = self.old_SPEC_PARAMS[key][p]
		else:
			if new_prob > self.old_prob:
				self.old_prob = new_prob
				self.accept_rate += 1
			else:
				for p in range(self.npix):
					for key in self.AMPLITUDES.keys():
						self.AMPLITUDES[key][p] = self.old_AMPLITUDES[key][p]
					for key in self.SPEC_PARAMS.keys():
						self.SPEC_PARAMS[key][p] = self.old_SPEC_PARAMS[key][p]                  
		
		
		# Storing samples
		for p in range(self.npix):
			for j,key in enumerate(self.AMPLITUDES.keys()):
				self.samples[i, p, j] = self.AMPLITUDES[key][p]
			for j,key in enumerate(self.SPEC_PARAMS.keys()):
				self.samples[i, p, 3 * len(self.components) + j] = self.SPEC_PARAMS[key][p]   




	def decorr_chain(self,i,pix,stokes,optimize=False):
		
		if not self.use_chains_covariance:  ## preconditioning run
			for k,key in enumerate(self.AMPLITUDES.keys()):
				self.old_AMPLITUDES[key] = self.AMPLITUDES[key].copy()
				if pix == -1:
					self.AMPLITUDES[key] += np.random.randn(self.npix) * self.STEPS[key]
				else:
					self.AMPLITUDES[key][pix] += np.random.randn() * self.STEPS[key][pix]	
		
			for k,key in enumerate(self.SPEC_PARAMS.keys()):
				self.old_SPEC_PARAMS[key] = self.SPEC_PARAMS[key].copy()
				if pix == -1:
					self.SPEC_PARAMS[key] += np.random.randn(self.npix) * self.STEPS[key]
				else:
					self.SPEC_PARAMS[key][pix] += np.random.randn() * self.STEPS[key][pix]  
		else:
			if pix == -1:
				samps = np.zeros((self.npars,self.npix))
				for p in range(self.npix):
					samps[:,p] = self.FULL_SQRTCOV @ np.random.randn(self.npars)
			else:
				samps = self.FULL_SQRTCOV @ np.random.randn(self.npars)

			for k,key in enumerate(self.AMPLITUDES.keys()):
				self.old_AMPLITUDES[key] = self.AMPLITUDES[key].copy()
				if pix == -1:
					self.AMPLITUDES[key] += samps[k]
				else:
					self.AMPLITUDES[key][pix] += samps[k]

			for k,key in enumerate(self.SPEC_PARAMS.keys()):
				self.old_SPEC_PARAMS[key] = self.SPEC_PARAMS[key].copy()
				if pix == -1:
					self.SPEC_PARAMS[key] += samps[3 * len(self.components) + k]
				else:
					self.SPEC_PARAMS[key][pix] += samps[3 * len(self.components) + k]


		
		#########-----  Check Proposal  -----#########
		
		new_prob = -0.5 * self.decorr_chi_sq(pix,stokes) + self.decorr_prior_cut(pix,stokes)
		
		if not optimize: 
			A = min(1., np.exp(new_prob - self.old_prob))
			if A >= np.random.uniform(0,1):  
				self.old_prob = new_prob      
				self.accept_rate += 1
			else:
				for p in range(self.npix):
					for key in self.AMPLITUDES.keys():
						self.AMPLITUDES[key][p] = self.old_AMPLITUDES[key][p]
					for key in self.SPEC_PARAMS.keys():
						self.SPEC_PARAMS[key][p] = self.old_SPEC_PARAMS[key][p]
		else:
			if new_prob > self.old_prob:
				self.old_prob = new_prob
				self.accept_rate += 1
			else:
				if pix == -1:
					for p in range(self.npix):
						for key in self.AMPLITUDES.keys():
							self.AMPLITUDES[key][p] = self.old_AMPLITUDES[key][p]
						for key in self.SPEC_PARAMS.keys():
							self.SPEC_PARAMS[key][p] = self.old_SPEC_PARAMS[key][p]
				else:
					for key in self.AMPLITUDES.keys():
						self.AMPLITUDES[key][pix] = self.old_AMPLITUDES[key][pix]
					for key in self.SPEC_PARAMS.keys():
						self.SPEC_PARAMS[key][pix] = self.old_SPEC_PARAMS[key][pix]                    
				
		# Storing samples
		if pix == -1:
			for p in range(self.npix):
				for j,key in enumerate(self.AMPLITUDES.keys()):
					self.samples[i, p, j] = self.AMPLITUDES[key][p]        
				for j,key in enumerate(self.SPEC_PARAMS.keys()):
					self.samples[i, p, 3 * len(self.components) + j] = self.SPEC_PARAMS[key][p]   
		else:
			for j,key in enumerate(self.AMPLITUDES.keys()):
				self.samples[i, pix, j] = self.AMPLITUDES[key][pix]        
			for j,key in enumerate(self.SPEC_PARAMS.keys()):
				self.samples[i, pix, 3 * len(self.components) + j] = self.SPEC_PARAMS[key][pix]   




	def metropolis(self, it=0, burn_in=0, optimize=False):
		
		if it != 0: self.it = it
		if burn_in != 0: self.burn_in = burn_in
		self.accept_rate = 0        
		
		### Computing initial Chi square
		print('\033[1m'+'Computing TOTAL initial Chi square'+'\033[0m')
		self.old_prob = -0.5 * self.chi_sq() + self.prior_cut()
		
	
		### Preconditioning if asked
		if self.use_chains_covariance:
			path = self.path + 'chain_sqrt_cov.dat'
			SQRTCOV = []
			with open(path,'r') as filename:
				for line in filename.readlines():
					SQRTCOV.append([float(el) for el in line[:-1].split('\t')])
				SQRTCOV = np.array(SQRTCOV)

			if np.shape(SQRTCOV) != (len(self.to_sample),len(self.to_sample)): 
				raise ValueError('Chain covariance preconditioner has wrong shape.')

			# Fill with unsampled parameters
			self.FULL_SQRTCOV = np.zeros((self.npars,self.npars))
			self.FULL_SQRTCOV[:len(self.to_sample),:len(self.to_sample)] = SQRTCOV.copy()


		### Actual Sampling
		self.old_AMPLITUDES, self.old_SPEC_PARAMS = {}, {}
		self.samples = np.empty(((self.it, self.npix, self.npars)))
		for i in tqdm(range(self.it), desc = 'MCMC chain (iterations)'):
			self.chain(i,optimize)
		
		print('Acceptance rate: '+str(self.accept_rate/self.it))
		print('Chain variances: ')
		for j,key in enumerate(self.AMPLITUDES.keys()):
			print('\t' + key + ':')
			for p in range(self.npix):
				print('\t\tpix. '+str(p)+':\t'+ str(np.var(self.samples[self.burn_in:, p, j]))) 
				self.variances[p,j] = np.var(self.samples[self.burn_in:, p, j])
		for j,key in enumerate(self.SPEC_PARAMS.keys()):
			print('\t' + key + ':')
			for p in range(self.npix):
				print('\t\tpix. '+str(p)+':\t'+ str(np.var(self.samples[self.burn_in:, p, 3 * len(self.components) + j])))
				self.variances[p,3 * len(self.components) + j] = np.var(self.samples[self.burn_in:, p,3 * len(self.components) + j])




	def decorr_metropolis(self, pix=-1, stokes = '', it=0, burn_in=0, optimize=False):
		
		if pix != -1 and self.pix_corr:
			raise ValueError("Pixels are correlated, use 'metropolis' function.") 
		if stokes != '' and self.IQU_corr:
			raise ValueError("IQU are correlated, use 'metropolis' function.")         
	
		if it != 0: self.it = it
		if burn_in != 0: self.burn_in = burn_in
		self.accept_rate = 0
		
		
		### Computing initial Chi square
		print('\033[1m'+'Computing initial Chi square'+'\033[0m')
		self.old_prob = -0.5 * self.decorr_chi_sq(pix,stokes) + self.decorr_prior_cut(pix,stokes)
		

		### Preconditioning if asked
		if self.use_chains_covariance:
			path = self.path + 'chain_sqrt_cov.dat'
			SQRTCOV = []
			with open(path,'r') as filename:
				for line in filename.readlines():
					SQRTCOV.append([float(el) for el in line[:-1].split('\t')])
				SQRTCOV = np.array(SQRTCOV)

			if np.shape(SQRTCOV) != (len(self.to_sample),len(self.to_sample)):
				raise ValueError('Chain covariance preconditioner has wrong shape.')

			# Fill with unsampled parameters
			self.FULL_SQRTCOV = np.zeros((self.npars,self.npars))
			self.FULL_SQRTCOV[:len(self.to_sample),:len(self.to_sample)] = SQRTCOV.copy()

		
		### Actual Sampling
		self.old_AMPLITUDES, self.old_SPEC_PARAMS = {}, {}
		
		## still allocate NPIX x ALL PARAMS slots but leave them empty
		self.samples = np.empty(((self.it, self.npix, self.npars)))
		self.variances = np.zeros((self.npix, self.npars))
		for i in tqdm(range(self.it), desc = 'MCMC chain (iterations)'):
			self.decorr_chain(i,pix,stokes,optimize)
		
		print('Acceptance rate: '+str(self.accept_rate/self.it))
		print('Chain variances: ')
		for j,key in enumerate(self.AMPLITUDES.keys()):
			print('\t' + key + ':')
			if pix == -1:
				for p in range(self.npix):
					print('\t\tpix. '+str(p)+':\t'+ str(np.var(self.samples[self.burn_in:, p, j])))
					self.variances[p,j] = np.var(self.samples[self.burn_in:, p, j])
			else:
				print('\t\tpix. '+str(pix)+':\t'+ str(np.var(self.samples[self.burn_in:, pix, j])))                    
				self.variances[pix,j] = np.var(self.samples[self.burn_in:, pix, j])
		for j,key in enumerate(self.SPEC_PARAMS.keys()):
			print('\t' + key + ':')
			if pix == -1:
				for p in range(self.npix):
					print('\t\tpix. '+str(p)+':\t'+ str(np.var(self.samples[self.burn_in:, p, 3 * len(self.components) + j])))
					self.variances[p,3 * len(self.components) + j] = np.var(self.samples[self.burn_in:, p, 3 * len(self.components) + j])
			else:
				print('\t\tpix. '+str(pix)+':\t'+ str(np.var(self.samples[self.burn_in:, pix, 3 * len(self.components) + j])))
				self.variances[pix,3 * len(self.components) + j] = np.var(self.samples[self.burn_in:, pix, 3 * len(self.components) + j])


      
	def plot_results(self, pix=0, scale='scal',png_title=''):
		
		## Reconstructed are in dashed line
		print('Solid: DATA; Dashed: Reconstructed by MODELS')
		print('PIXEL '+str(pix)+'\n\n')
		colors = ['green','red','blue','orange','purple']
		
		fig1, ax1 = plt.subplots(1,3,figsize=(20,8))
		fig2, ax2 = plt.subplots(1,3,figsize=(20,8))
		
		for i in range(3):
			stokes = ['I','Q','U'][i]
			models = []
			for j,comp in enumerate(self.components):
				data_ = []; model_em = []
				for f in self.nus:
					data_.append(self.data[comp+'_'+stokes+'_'+str(f)][pix])
					model_em.append(self.emission(comp,f,pix,stokes))
			
				ax1[i].plot(self.nus,data_,c=colors[j],label=comp, linestyle='-')
				ax1[i].plot(self.nus,model_em,c=colors[j], linestyle='--')
				models.append(model_em.copy())
			ax1[i].set_xlabel('Frequencies [GHz]',fontsize=15)
			ax1[i].set_ylabel(stokes+' amplitude [$\mu K_{CMB}$]',fontsize=15)
			ax1[i].set_title(stokes,fontsize=20)
			ax1[i].legend(fontsize=15)
			
			## Ratios
			for j,comp in enumerate(self.components):
				data_ = []
				for f in self.nus:
					data_.append(self.data[comp+'_'+stokes+'_'+str(f)][pix])
				
				if np.array(data_).all() != 0:
					ax2[i].plot(self.nus,np.array(models[j])/np.array(data_),c=colors[j],label=comp)
			ax2[i].set_xlabel('Frequencies [GHz]',fontsize=15)
			ax2[i].set_ylabel('Model/Data',fontsize=15)
			ax2[i].set_title(stokes,fontsize=20)
			ax2[i].legend(fontsize=15)
		
		
		if scale=='log':
			ax1[0].set_xscale('log'); ax1[0].set_yscale('log')
			ax1[1].set_xscale('log'); ax1[1].set_yscale('log')
			ax1[2].set_xscale('log'); ax1[2].set_yscale('log')     

		fig1.savefig(png_title+'_top.png',dpi=300)
		fig2.savefig(png_title+'_bottom.png',dpi=300)





