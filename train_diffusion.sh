# DiT-S: 30M, DiT-B: 150M, DiT-L: 450M
# d_model: 768    # 384, 768, 1024
# nhead: 12        # 6, 12, 16
# num_layers: 12  # 12, 12, 24

d_x=9 ##(D+1)
kl=0.0001

# num_layers=12
# d_model=768
# nhead=12
# name="DiT-B_vae_latent@${d_x}_kl@${kl}"

num_layers=12
d_model=384
nhead=6
name="DiT-S_vae_latent@${d_x}_kl@${kl}"

# Path to script and working directory
application="python src/train_diffusion.py"
# workdir="./"

options="trainer=gpu logger=wandb name=$name \
    ++diffusion_module.denoiser.num_layers=$num_layers \
    ++diffusion_module.denoiser.d_model=$d_model \
    ++diffusion_module.denoiser.nhead=$nhead \
    ++diffusion_module.denoiser.d_x=$d_x"

CMD="HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 $application $options"


# options="trainer=ddp logger=wandb name=$name \
#     ++diffusion_module.denoiser.num_layers=$num_layers \
#     ++diffusion_module.denoiser.d_model=$d_model \
#     ++diffusion_module.denoiser.nhead=$nhead \
#     ++diffusion_module.denoiser.d_x=$d_x"

# CMD="HYDRA_FULL_ERROR=1 $application $options"

# Go to working directory


echo "Running command:"
echo "$CMD"
echo

# Execute
eval $CMD
