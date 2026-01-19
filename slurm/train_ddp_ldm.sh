"""Copyright (c) Meta Platforms, Inc. and affiliates."""

#!/bin/bash
#! Name of the job:
#SBATCH -J ldm
#SBATCH -o slurm/logs/train%j.out # File to which STDOUT will be written
#SBATCH -e slurm/logs/train%j.err # File to which STDERR will be written

#! Partition:
#SBATCH -p mypartition

#! How many whole compute nodes should be allocated?
#SBATCH --nodes=1

#! How many total CPU tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks-per-node=8

#! Specify the number of GPUs per node (between 1 and 8).
#SBATCH --gpus-per-node=8

#! Specify the number of CPUs for your job. You should always allocate only 10 CPUs per requested GPU.
#! Requesting less than 10 CPUs per GPU will make the GPU efficiency low.
#SBATCH --cpus-per-task=10

#! How much memory is required? (default to 64GB if not specified)
#SBATCH --mem=64GB

#! How much wallclock time will be required? (at most 72:00:00 in general)
#SBATCH --time=72:00:00

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
application="srun python /private/home/myuser/all-atom-diffusion-transformer/src/train_diffusion.py"

#! Set autoencoder_ckpt and hparams in configs/diffusion_module/ldm.yaml, or below:
d_x=8  # 4 / 8
kl=0.00001  # 0.0001 / 0.00001
num_layers=12  # 12, 12, 24
d_model=768  # 384, 768, 1024
nhead=12  # 6, 12, 16
# for DiT-L, add to options: ++data.datamodule.batch_size.train=32 ++trainer.accumulate_grad_batches=8

#! (for logging purposes)
name="DiT-B__vae_latent@${d_x}_kl@${kl}_joint"

#! Run options for the application:
options="trainer=ddp logger=wandb name=$name ++diffusion_module.denoiser.num_layers=$num_layers ++diffusion_module.denoiser.d_model=$d_model ++diffusion_module.denoiser.nhead=$nhead ++diffusion_module.denoiser.d_x=$d_x"

#! Work directory (i.e. where the job will run):
workdir="/private/home/myuser/all-atom-diffusion-transformer/"

CMD="HYDRA_FULL_ERROR=1 $application $options"

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
