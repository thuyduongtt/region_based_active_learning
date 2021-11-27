#!/bin/bash

#SBATCH -J al    # Job Name

## Leave these values as they are unless you know what you are doing
#SBATCH --ntasks=1             # Tasks
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks-per-core=2    # Number of processes per CPU code
#SBATCH --cpus-per-task=1      # Number of CPUs per task?

## Adjust to your needs
#SBATCH --mem=25GB             # Memory limit per node
#SBATCH --time=00:02:00        # Expected maximum run time
#SBATCH --partition=standard   # This is for testing on CPU

## Job-Status per Mail
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.tranthi@campus.tu-berlin.de

ulimit -u 512

source activate act
python3 tf.py