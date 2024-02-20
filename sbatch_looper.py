import os

print('NOISE REALIZATION: ')
n = input()
print('NUMBER OF CHAINS: ')
c = int(input())

for i in range(c):
    stream = open('dummy_sbatch.sh','w')
    stream.write('#!/bin/bash\n')
    stream.write('#SBATCH --account=cmbgroup\n')
    stream.write('#SBATCH --chdir=/home/users/luca.zapelli/cmbgroup/users/luca.zapelli/comm_2.5_validation\n')
    stream.write('#SBATCH --time=1:00:00\n')
    stream.write('#SBATCH --partition=galileo\n')
    stream.write('#SBATCH --nodes=1\n')
    stream.write('#SBATCH --ntasks=1\n')
    stream.write('#SBATCH --cpus-per-task=1\n')
    stream.write('#SBATCH --mem=1G\n')
    stream.write('#SBATCH --job-name=valid'+str(i)+'\n')
    stream.write('#SBATCH --output=slurm_%j.log\n')
    stream.write('#SBATCH --error=error_%j.log\n')
    stream.write('#SBATCH --mail-user=luca.zapelli@unimi.it\n')
    stream.write('#SBATCH --mail-type=END,FAIL\n\n')

    stream.write('python 3_my_component_separation.py '+n+' '+str(i+1))
    stream.close()

    os.system('sbatch dummy_sbatch.sh')
    os.system('rm dummy_sbatch.sh')
