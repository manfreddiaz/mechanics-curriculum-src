#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=8                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 12 GB of RAM
#SBATCH -o /network/scratch/d/diazcabm/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate metacurriculum

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python main.py +xper=minatar_dqn_ordered ++thread_pool.size=16 ++run.outdir=$SLURM_TMPDIR/logs/static/

# 5. Copy whatever you want to save on $SCRATCH
cp -R $SLURM_TMPDIR/static/ /network/scratch/d/diazcabm/
