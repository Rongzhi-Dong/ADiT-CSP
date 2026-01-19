d_x=8
kl=0.00001
num_layers=12
d_model=768
nhead=12

name="DiT-B_vae_latent@${d_x}_kl@${kl}_joint"

# Path to script and working directory
application="python src/train_diffusion.py"
# workdir="./"

options="trainer=ddp logger=wandb name=$name \
    ++diffusion_module.denoiser.num_layers=$num_layers \
    ++diffusion_module.denoiser.d_model=$d_model \
    ++diffusion_module.denoiser.nhead=$nhead \
    ++diffusion_module.denoiser.d_x=$d_x"

CMD="HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 $application $options"

# Go to working directory


echo "Running command:"
echo "$CMD"
echo

# Execute
eval $CMD
