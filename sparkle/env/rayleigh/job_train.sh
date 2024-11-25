#!/bin/bash
#
#SBATCH --job-name=opt_rayleigh
#SBATCH --output=opt_rayleigh.txt
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=2-00:00:00
#
source /home/jviquerat/scratch/sparkle/venv/bin/activate
module load openmpi/4.1.1
mpirun -n 10 spk --train cmaes.json
