"""Copyright (c) Meta Platforms, Inc. and affiliates."""

#!/bin/bash
#! Name of the job:
#SBATCH -J sweep
#SBATCH -o slurm/logs/sweep%j.out # File to which STDOUT will be written
#SBATCH -e slurm/logs/sweep%j.err # File to which STDERR will be written

#! Partition:
#SBATCH -p mypartition

# Array of jobs for sweep
#SBATCH --array=0-35

#! How many whole compute nodes should be allocated?
#SBATCH --nodes=1

#! How many total CPU tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks-per-node=1

#! Specify the number of GPUs per node (between 1 and 8).
#SBATCH --gpus-per-node=1

#! Specify the number of CPUs for your job. You should always allocate only 10 CPUs per requested GPU.
#! Requesting less than 10 CPUs per GPU will make the GPU efficiency low.
#SBATCH --cpus-per-task=10

#! How much memory is required? (default to 64GB if not specified)
#SBATCH --mem=64GB

#! How much wallclock time will be required? (at most 72:00:00 in general)
#SBATCH --time=12:00:00

#! sbatch directives end here (put any additional directives above this line)

############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
###./etc/profile.d/modules.sh                 # Enables the module command
###module purge                               # Removes all modules still loaded

#! Insert additional module load commands after this line if needed:
###module load ...
###module list

#! Load python environment
source /private/home/myuser/.bashrc
mamba activate myenv

############################################################

#! Full path to application executable:
application="wandb agent wandb-entity/all-atom-diffusion-transformer/sweep-id"

#! Run options for the application:
options="--count 1"

#! Work directory (i.e. where the job will run):
workdir="/private/home/myuser/all-atom-diffusion-transformer/"

CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
