import lib.Utils as utils
import numpy as np
import healpy as hp
import scipy.linalg as la
import sys
import time

class Instrument:

	def __init__(self,pars,build_cov=True):

		self.nus = np.array(pars['nus'])*1e9
		self.nbands = len(self.nus)
		self.bwidths = np.array(pars['bwidths'])*1e9
		if len(self.bwidths) != self.nbands:
			raise ValueError('Incorrect number of bandwidths')
		self.dnu_nu = self.bwidths/self.nus
		edges_min = self.nus * (1 - self.dnu_nu/2)	
		edges_max = self.nus * (1 + self.dnu_nu/2)
		self.edges = np.array([[edges_min[i], edges_max[i]] \
				      for i in range(self.nbands)])
		if self.bwidths.any() == 0:
			self.edges = self.nus.copy()
		self.beam_fwhm = np.array(pars['fwhm'])
		if len(self.beam_fwhm) != self.nbands:
			raise ValueError('Incorrect number of beams')
		self.max_nside()
		if pars['nside'] > self.max_Nside:	
			raise ValueError('Requested resolution is too high')
		self.nside = pars['nside']
		hp_fwhm = hp.nside2resol(nside=self.nside, arcmin=True)
		self.fwhm = np.array([max(pars['fwhm'][i],hp_fwhm) \
				     for i in range(self.nbands)])
		self.npixs = hp.nside2npix(self.nside)
		self.center_GAL = np.array(pars['center_GAL'])
		if len(self.center_GAL) != 2:
			raise ValueError('Incorrect galactic coordinates')
		self.fsky = pars['fsky']
		self.coverage()
		
		self.rng_pix_scatter = 1 + np.random.randn(self.npix)/100
		self.noise_rms = np.ones(((self.nbands,3,self.npix)))
		for i in range(self.nbands):
			self.noise_rms[i,0] *= pars['rms_I_ukarcmin'][i]
			self.noise_rms[i,1] *= pars['rms_Q_ukarcmin'][i]
			self.noise_rms[i,2] *= pars['rms_U_ukarcmin'][i]
			self.noise_rms[i] /= self.fwhm[i]
		
			for j in range(3):
				self.noise_rms[i,j] *= self.rng_pix_scatter
		
		
		if build_cov:
			
			self.QUBIC_covar = pars['QUBIC_covar']
			
			self.pix_corr_lvl = pars['pix_corr_lvl']
			if self.pix_corr_lvl < 0 or self.pix_corr_lvl > 1:
				raise ValueError('Incorrect pixel correlation level')
				
			if not self.QUBIC_covar:
				self.IQU_corr_lvl = pars['IQU_corr_lvl']
				if self.IQU_corr_lvl < 0 or self.IQU_corr_lvl > 1:
					raise ValueError('Incorrect IQU correlation level')
				self.nu_corr_lvl = pars['nu_corr_lvl']
				if self.nu_corr_lvl < 0 or self.nu_corr_lvl > 1:
					raise ValueError('Incorrect frequency correlation level')
			else:
				self.real_corr_avg, self.perturb = self.realistic_corr()
			
			self.cov_submatrices, self.corr_submatrices = {}, {}
			for i,f1 in enumerate(self.nus):
				for j,f2 in enumerate(self.nus):
					if str(f2)+'_'+str(f1) not in self.cov_submatrices.keys():
						self.set_covariance(f1,f2)
			
			self.build_util_covs()
			
			spinner = utils.spinning_cursor()
			while self.def_pos != 0:
				sys.stdout.write(next(spinner))
				sys.stdout.flush()
				time.sleep(0.1)
				sys.stdout.write('\b')
			
				self.cov_submatrices, self.corr_submatrices = {}, {}
				for i,f1 in enumerate(self.nus):
					for j,f2 in enumerate(self.nus):
						if str(f2)+'_'+str(f1) not in self.cov_submatrices.keys():
							self.set_covariance(f1,f2)
				
				self.build_util_covs()
			
			print('Definite positive covariance successfully built!')
			if pars['print_path'] != '':
				self.print_covs(pars['print_path'])
		
	
	
	def max_nside(self):
		N = 1
		reso = hp.pixelfunc.nside2resol(nside=N,arcmin=True)
		while reso > max(self.beam_fwhm):
			N *= 2
			reso = hp.pixelfunc.nside2resol(nside=N,arcmin=True)

		self.max_Nside = int(N/2)
		

	def coverage(self):
		uvcenter = np.array(hp.ang2vec(self.center_GAL[0], self.center_GAL[1], lonlat=True))
		uvpix = np.array(hp.pix2vec(self.nside, np.arange(self.npixs)))
		ang = np.arccos(np.dot(uvcenter, uvpix))
		indices = np.argsort(ang)
		okpix = ang < -1
		okpix[indices[0:int(self.fsky * self.npixs)]] = True
		cov = np.zeros(self.npixs)
		cov[okpix] = 1
		self.mask = cov.copy()
		self.patch_pixels = []
		for i,p in enumerate(self.mask):
			if p>0:
				self.patch_pixels.append(int(i))
		self.npix = len(self.patch_pixels)



	def realistic_corr(self):
		corr = {}
		corr['II'] = np.array([[      1.,-0.38961,-0.25649],
				       [-0.38961,      1.,-0.28409],
				       [-0.25649,-0.28409,      1.]])
		corr['QI'] = np.array([[-0.06818, 0.02922,-0.11201],
				       [ 0.06494, 0.05195, 0.07792],
				       [ 0.18182, 0.02273,-0.17208]])
		corr['QQ'] = np.array([[      1.,-0.62338,-0.12662],
				       [-0.62338,      1.,-0.42857],
				       [-0.12662,-0.42857,      1.]])
		corr['UI'] = np.array([[-0.07143,-0.15260,-0.03571],
				       [ 0.04545, 0.08117, 0.03896],
				       [-0.01299, 0.10065,-0.01299]])
		corr['UQ'] = np.array([[ 0.19156, 0.00984,-0.05844],
				       [-0.17857, 0.06818, 0.04221],
				       [-0.20454, 0.17208, 0.01948]])
		corr['UU'] = np.array([[      1.,-0.45455,-0.29221],
				       [-0.45455,      1.,-0.13312],
				       [-0.29221,-0.13312,      1.]])

		PIX_real_corr = np.hstack((np.vstack((corr['II'],corr['QI'].T,corr['UI'].T)),
				np.vstack((corr['QI'],corr['QQ']  ,corr['UQ'].T)),
				np.vstack((corr['UI'],corr['UQ']  ,corr['UU']  ))))

		corr['II'] = np.array([[      1.,-0.43750,-0.27302],
				       [-0.43750,      1.,-0.43421],
				       [-0.27302,-0.43421,      1.]])
		corr['QI'] = np.zeros((3,3))
		corr['QQ'] = np.array([[      1.,-0.43092,-0.32566],
				       [-0.43092,      1.,-0.38487],
				       [-0.32566,-0.38487,      1.]])
		corr['UI'] = np.zeros((3,3))
		corr['UQ'] = np.zeros((3,3))
		corr['UU'] = np.array([[      1.,-0.42763,-0.32237],
				       [-0.42763,      1.,-0.37829],
				       [-0.32237,-0.37829,      1.]])

		AVG_real_corr = np.hstack((np.vstack((corr['II'],corr['QI'].T,corr['UI'].T)),
				np.vstack((corr['QI'],corr['QQ']  ,corr['UQ'].T)),
				np.vstack((corr['UI'],corr['UQ']  ,corr['UU']  ))))

		PERTURB_real = np.abs(PIX_real_corr - AVG_real_corr)
		return AVG_real_corr, PERTURB_real


	def gen_pix_corr(self):
		corr = self.real_corr_avg.copy()
		for i in range(9):
			for j in range(i,9):
				corr[i,j] += np.random.normal(0,self.perturb[i,j])
				corr[j,i] = corr[i,j]
		return corr


	def gen_nu_corr(self,f1,f2=0):
		if f2 == 0:
			f2=f1
		f1_ind = list(self.nus).index(f1)
		f2_ind = list(self.nus).index(f2)	
		subs = np.zeros(((6,self.npix,self.npix)))
		
		for i in range(self.npix):
			for j in hp.get_all_neighbours(self.nside,self.patch_pixels[i]):
				if j in self.patch_pixels:
					j = self.patch_pixels.index(j)
					subs[:,i,j] = self.pix_corr_lvl * np.random.normal(1,0.1)
					subs[:,j,i] = subs[:,i,j]

		for i,p in enumerate(self.patch_pixels):
			SIM_corr = self.gen_pix_corr()
			subs[0,i,:] *= SIM_corr[f1_ind,f2_ind]
			subs[0,:,i] *= SIM_corr[f1_ind,f2_ind]
			subs[0,i,i]  = SIM_corr[f1_ind,f2_ind]
			
			subs[1,i,:] *= SIM_corr[f1_ind+3,f2_ind]
			subs[1,:,i] *= SIM_corr[f1_ind+3,f2_ind]
			subs[1,i,i]  = SIM_corr[f1_ind+3,f2_ind]
			
			subs[2,i,:] *= SIM_corr[f1_ind+3,f2_ind+3]
			subs[2,:,i] *= SIM_corr[f1_ind+3,f2_ind+3]
			subs[2,i,i]  = SIM_corr[f1_ind+3,f2_ind+3]
			
			subs[3,i,:] *= SIM_corr[f1_ind+6,f2_ind]
			subs[3,:,i] *= SIM_corr[f1_ind+6,f2_ind]
			subs[3,i,i]  = SIM_corr[f1_ind+6,f2_ind]
			
			subs[4,i,:] *= SIM_corr[f1_ind+6,f2_ind+3]
			subs[4,:,i] *= SIM_corr[f1_ind+6,f2_ind+3]
			subs[4,i,i]  = SIM_corr[f1_ind+6,f2_ind+3]
		
			subs[5,i,:] *= SIM_corr[f1_ind+6,f2_ind+6]
			subs[5,:,i] *= SIM_corr[f1_ind+6,f2_ind+6]
			subs[5,i,i]  = SIM_corr[f1_ind+6,f2_ind+6]

		corr = np.hstack((np.vstack((subs[0],subs[1],subs[3])),
                                  np.vstack((subs[1],subs[2],subs[4])),
                                  np.vstack((subs[3],subs[4],subs[5]))))
		return corr


	def set_covariance(self, f1, f2=0):
		if f2==0:
			f2=f1
		f1_ind = list(self.nus).index(f1)
		f2_ind = list(self.nus).index(f2)
	
		if str(f2)+'_'+str(f1) not in self.cov_submatrices.keys():
			n = self.npix
			self.corr_submatrices[str(f1)+'_'+str(f2)] = np.zeros((3*n,3*n))
			self.cov_submatrices[str(f1)+'_'+str(f2)] = np.zeros((3*n,3*n))
		else:
			return
	
		if np.abs(f1_ind - f2_ind) > 1:
			return
	
		if self.QUBIC_covar:
			self.corr_submatrices[str(f1)+'_'+str(f2)] += self.gen_nu_corr(f1,f2)
		else:
			# pix correlation (w/ scatter)
			corr_submatrices = np.zeros(((6,n,n)))
			for i in range(n):
				for j in hp.get_all_neighbours(self.nside,self.patch_pixels[i]):
					if j in self.patch_pixels:
						j = self.patch_pixels.index(j)
						corr_submatrices[:,i,j] = self.pix_corr_lvl * np.random.normal(1,0.01) * (-1)**np.random.randint(2)
						corr_submatrices[:,j,i] = corr_submatrices[:,i,j]                

			corr_submatrices[:] += np.eye(n)

			# IQU correlation (w/ scatter)
			corr_submatrices[1] *= self.IQU_corr_lvl * np.random.normal(1,0.01) * (-1)**np.random.randint(2)
			corr_submatrices[3] *= self.IQU_corr_lvl * np.random.normal(1,0.01) * (-1)**np.random.randint(2)
			corr_submatrices[4] *= self.IQU_corr_lvl * np.random.normal(1,0.01) * (-1)**np.random.randint(2)

			# freq correlation
			if f1 != f2:
				freq_corr = self.nu_corr_lvl * np.random.normal(1,0.01) * (-1)**np.random.randint(2)
				corr_submatrices[:] *= freq_corr
                
			# reshape matrix
			self.corr_submatrices[str(f1)+'_'+str(f2)] = np.hstack((np.vstack((corr_submatrices[0],corr_submatrices[1],corr_submatrices[3])),\
										np.vstack((corr_submatrices[1],corr_submatrices[2],corr_submatrices[4])),\
										np.vstack((corr_submatrices[3],corr_submatrices[4],corr_submatrices[5]))))
			del corr_submatrices
		
		self.cov_submatrices[str(f1)+'_'+str(f2)] = self.corr_submatrices[str(f1)+'_'+str(f2)].copy()
		for i in range(3):
			for j in range(3):
				for p in range(n):
					self.cov_submatrices[str(f1)+'_'+str(f2)][i*n+p,j*n:(j+1)*n] *= self.noise_rms[f1_ind,i,p]
					self.cov_submatrices[str(f1)+'_'+str(f2)][i*n:(i+1)*n,j*n+p] *= self.noise_rms[f2_ind,j,p]
	
	
	def build_util_covs(self):
		n = self.npix
		self.inst_covariance = np.zeros((3*n*self.nbands,3*n*self.nbands))
		for i,f1 in enumerate(self.nus):
			for j,f2 in enumerate(self.nus):
				if str(f1)+'_'+str(f2) not in self.cov_submatrices.keys():
					continue
				
				self.inst_covariance[i*3*n:(i+1)*3*n,j*3*n:(j+1)*3*n] = self.cov_submatrices[str(f1)+'_'+str(f2)].copy()
				if j!=i:
					self.inst_covariance[j*3*n:(j+1)*3*n,i*3*n:(i+1)*3*n] = self.cov_submatrices[str(f1)+'_'+str(f2)].copy()
			
		self.def_pos=0
		for eig in la.eig(self.inst_covariance)[0]:
			if eig.real <= 0 or eig.imag != 0:
				self.def_pos = 1
				return
			


	def print_covs(self,path):
		for f1 in self.nus:
			for f2 in self.nus:
				if str(f1)+'_'+str(f2) in self.cov_submatrices.keys():
					cov = self.cov_submatrices[str(f1)+'_'+str(f2)].copy()
					stream = open(path+'TRUE_cov_'+str(f1)+'_'+str(f2)+'.txt','w')
					for i in range(3*self.npix):
						for j in range(3*self.npix):
							stream.write(str(cov[i,j]))
							if j<3*self.npix-1:
								stream.write('\t')
							else:
								stream.write('\n')
					stream.close()



