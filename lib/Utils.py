import numpy as np
import scipy.linalg as la
import os
import sys

def spinning_cursor():
	while True:
		for cursor in '|/-\\':
			yield cursor


def generate_corr_vars(cov, MU=0., seed=0):
	size = np.shape(cov)[0]

	if cov.any() == 0:
		return np.zeros(size)
	L = la.cholesky(cov,lower=True)

	if seed != 0:
		np.random.seed(seed)
	x = np.random.randn(size)
	mu = np.ones(size) * MU
	y = mu + L @ x
	return y



def build_mcmc_covs(nus,npix,path_in,path_out,pix_dense=False,IQU_dense=False,nu_dense=False,print_=False,return_=False):
	nus = [float(x)*1e9 for x in nus]
	mcmc_full_cov = np.zeros((3*npix*len(nus),3*npix*len(nus)))
	subcov_names = []
	for i,f1 in enumerate(nus):
		for j,f2 in enumerate(nus):
			if str(f2)+'_'+str(f1) in subcov_names:
				continue
			
			mcmc_subcov = np.zeros((3*npix,3*npix))
			covname = str(f1)+'_'+str(f2)
			subcov_names.append(covname)
			if f1!=f2 and not nu_dense:
				pass	
			else:
				filename = path_in+'TRUE_cov_'+covname+'.txt'
				stream = open(filename,'r')
				text = stream.readlines()
				stream.close()
				mcmc_subcov = []
				for line in text:
					mcmc_subcov.append([float(el) for el in line[:-1].split('\t')])
				mcmc_subcov = np.array(mcmc_subcov)
				if not IQU_dense:
					for k in range(3):
						if k==0:
							mcmc_subcov[:npix,npix:] *= 0
						if k==1:
							mcmc_subcov[npix:2*npix,:npix] *= 0 
							mcmc_subcov[npix:2*npix,2*npix:3*npix] *= 0 
						if k==2:
							mcmc_subcov[2*npix:3*npix,:2*npix] *= 0 
				if not pix_dense:
					for k1 in range(3):
						for k2 in range(3):
							for p1 in range(npix):
								for p2 in range(npix):
									if p1!=p2:
										mcmc_subcov[k1*npix+p1,k2*npix+p2]=0

			
			mcmc_full_cov[3*npix*i:3*npix*(i+1),3*npix*j:3*npix*(j+1)] = mcmc_subcov.copy()
			if j!=i:
				mcmc_full_cov[3*npix*j:3*npix*(j+1),3*npix*i:3*npix*(i+1)] = mcmc_subcov.copy()
				
	if mcmc_full_cov.any()==0:
		mcmc_inv_cov = mcmc_full_cov.copy()
		mcmc_sqrt_cov = mcmc_inv_cov.copy()
	else:
		mcmc_inv_cov = la.inv(mcmc_full_cov.copy())
		mcmc_sqrt_cov = la.cholesky(mcmc_inv_cov, lower=True)

	if return_:
		return mcmc_inv_cov

	if print_:
		if not os.path.exists(path_out):
			os.system('mkdir '+path_out)

		for i,f1 in enumerate(nus):
			for j,f2 in enumerate(nus):
				if str(f1)+'_'+str(f2) not in subcov_names:
					continue

				cov = mcmc_inv_cov[3*npix*i:3*npix*(i+1),3*npix*j:3*npix*(j+1)].copy()
				stream = open(path_out+'Inv_cov_'+str(f1)+'_'+str(f2)+'.txt','w')
				stream.write(str(3*npix)+'\n')
				stream.write('1\n')
				stream.write('1\n')
				for p1 in range(3*npix):
					for p2 in range(3*npix):
						stream.write(str(cov[p1,p2]))
						if p2 < 3*npix-1:
							stream.write('\t')
						else:
							stream.write('\n')
				stream.close()
					

				cov = mcmc_sqrt_cov[3*npix*i:3*npix*(i+1),3*npix*j:3*npix*(j+1)].copy()
				stream = open(path_out+'Sqrt_Inv_cov_'+str(f1)+'_'+str(f2)+'.txt','w')
				stream.write(str(3*npix)+'\n')
				stream.write('1\n')
				stream.write('1\n')
				for p1 in range(3*npix):
					for p2 in range(3*npix):
						stream.write(str(cov[p1,p2]))
						if p2 < 3*npix-1:
							stream.write('\t')
						else:
							stream.write('\n')
				stream.close()
	
		
