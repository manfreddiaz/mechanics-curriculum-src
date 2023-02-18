#!/bin/bash

#SBATCH --partition=long-cpu-grace                           # Ask for unkillable job
#SBATCH --cpus-per-task=12                                # Ask for 2 CPUs
#SBATCH --gres=gpu:0                                     # Ask for 1 GPU
#SBATCH --mem=64G                                        # Ask for 10 GB of RAM
#SBATCH -o /network/scratch/d/diazcabm/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load anaconda/3

# 2. Load your environment
conda activate metacurriculum

# 3. Copy your dataset on the compute node
# cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python main.py --outdir $SLURM_TMPDIR/static 

# 5. Copy whatever you want to save on $SCRATCH
cp -R $SLURM_TMPDIR/static/ /network/scratch/d/diazcabm/
