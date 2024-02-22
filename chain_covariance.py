import numpy as np
import scipy.linalg as la
import os
import json

print('NOISE REALIZATION: ')
noise_real = input()

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



## Collect samples
CHAINS = {'pars': []}
path = map_params['print_path'] + cov_path + 'precond_REAL_' + noise_real +'/'
nchains = 0
for file in os.listdir(path):
    if file[:9] == 'Chain_log':
        nchains+=1

for n in range(nchains):
    CHAINS[str(n)] = {'samples': []}
    with open(path+'Chain_log_n'+str(n+1)+'.txt','r') as filename:
        f = filename.read().split('\n')
    iterations = int(f[0][12:])
    burn_in = int(f[1][9:])
    for i,line in enumerate(f[4:]):
        if n == 0 and i%(iterations+6) == 0:
            CHAINS['pars'].append(line)
        if i%(iterations+6) > 4 and i%(iterations+6) < iterations+5:
            CHAINS[str(n)]['samples'].append(float(line))
    CHAINS[str(n)]['samples'] = np.array(CHAINS[str(n)]['samples'])

CHAINS['pars'] = CHAINS['pars'][:-1]


PARS = CHAINS['pars'][:-2] # If spectral parameters are not sampled, exclude them to avoid not positive definite covariance

nsamps = int(mcmc_params['iterations'] - mcmc_params['burn-in'])
samples = np.empty((len(PARS), nchains*nsamps))
for par in range(len(PARS)):
    for n in range(nchains):
        for s in range(nsamps):
                samples[par,n*nsamps + s] = CHAINS[str(n)]['samples'][par*mcmc_params['iterations'] + mcmc_params['burn-in'] + s]

## Print chain sqrt covariance
Cov = np.zeros((len(PARS),len(PARS)))
for par1 in range(len(PARS)):
    for par2 in range(len(PARS)):
        if par1 == par2:
            Cov[par1,par2] = np.var(samples[par1,:])
        else:
            for x,y in zip(samples[par1,:],samples[par2,:]):
                Cov[par1,par2] += (x - np.mean(samples[par1,:])) *\
                                  (y - np.mean(samples[par2,:])) / (nchains*nsamps-1)

sqrtcov = la.cholesky(Cov,lower=True)
with open(path+'chain_sqrt_cov.dat','w') as filename:
    for row in range(len(PARS)):
        for col in range(len(PARS)):
            filename.write(str(sqrtcov[row][col]))
            if col<len(PARS)-1:
                filename.write('\t')
        filename.write('\n')


