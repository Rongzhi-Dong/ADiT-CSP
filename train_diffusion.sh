
dataset=mpts_52  #   mp_20 / perov_5 / carbon_24 / mpts_52
d_x=8 # 4 / 6
loss_weight_kl=0.0001  # 0.0001 / 0.00001

name="${dataset}_cond_DiT-S_vae_latent@${d_x}_kl@${loss_weight_kl}"



# Path to script and working directory
# ## carbon_24
# application="python src/train_diffusion.py\
#     data=$dataset \
#     diffusion_module.autoencoder_ckpt=/home/rongzhid/ADiT-CSP/logs/train_autoencoder/runs/carbon_24_cond_vae_latent@8_kl@0.0001_2026-03-12_17-06-59/checkpoints/vae-epoch@99-step@2300-val_match_rate@0.9355.ckpt"
    

# ## preov_5
# application="python src/train_diffusion.py\
#     data=$dataset \
#     diffusion_module.autoencoder_ckpt=/home/rongzhid/ADiT-CSP/logs/train_autoencoder/runs/perov_5_cond_vae_latent@8_kl@0.0001_2026-03-12_17-07-55/checkpoints/vae-epoch@49-step@2200-val_match_rate@1.0000.ckpt"

## mpts_52
application="python src/train_diffusion.py\
    data=$dataset \
    diffusion_module.autoencoder_ckpt=/home/rongzhid/ADiT-CSP/logs/train_autoencoder/runs/mpts_52_cond_vae_latent@8_kl@0.0001_2026-03-12_17-06-29/checkpoints/vae-epoch@99-step@9400-val_match_rate@0.9849.ckpt"

options="trainer=gpu logger=wandb name=$name"

CMD="HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 $application $options"
# CMD="HYDRA_FULL_ERROR=1  $application $options"

# Go to working directory


echo "Running command:"
echo "$CMD"
echo

# Execute
eval $CMD
