#!/bin/bash

#SBATCH -J regional_active_learning    # Job Name

## Leave these values as they are unless you know what you are doing
# SBATCH --ntasks=1             # Tasks
# SBATCH --nodes=1              # Number of nodes
# SBATCH --ntasks-per-core=2    # Number of processes per CPU code
# SBATCH --cpus-per-task=1      # Number of CPUs per task?

## Adjust to your needs
# SBATCH --mem=25GB             # Memory limit per node
# SBATCH --time=50:00:00        # Expected maximum run time
# SBATCH --partition=standard   # This is for testing on CPU

## Job-Status per Mail
# SBATCH --mail-type=ALL
# SBATCH --mail-user=d@dttt.io

ulimit -u 512
rm -d output/ -r
rm -d Exp_Stat/Method_D_Stage_3_Version_0/ -r

echo "source ..."
source ~/anaconda3/bin/activate ~/anaconda3/envs/act
echo "module load nvidia/cuda/10.0"
module load nvidia/cuda/10.0
echo "start program ..."
STORAGE_DEFAULT_DIRECTORY="$PWD/storage/" python3 Train_Active_Full_Im.py --stage 3 --gpu_only