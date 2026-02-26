#! Set hparams in configs/autoencoder_module/vae.yaml, or below:
latent_dim=8  # 4 / 8
loss_kl=0.00001  # 0.0001 / 0.00001

#! (for logging purposes)
latent_str="latent@${latent_dim}"
kl_str="kl@${loss_kl}"
name="vae_${latent_str}_${kl_str}"

# Path to script and working directory

application="python src/train_autoencoder.py"

# workdir="./"

# options="trainer=ddp logger=wandb name=$name \
#     ++autoencoder_module.latent_dim=$latent_dim \
#     ++autoencoder_module.loss_weights.loss_kl.mp20=$loss_kl \
#     ++autoencoder_module.loss_weights.loss_kl.qm9=$loss_kl"


options="trainer=gpu logger=wandb name=$name \
    ++autoencoder_module.latent_dim=$latent_dim \
    ++autoencoder_module.loss_weights.loss_kl.mp20=$loss_kl"

CMD="HYDRA_FULL_ERROR=1 $application $options"

# Go to working directory


echo "Running command:"
echo "$CMD"
echo

# Execute
eval $CMD
