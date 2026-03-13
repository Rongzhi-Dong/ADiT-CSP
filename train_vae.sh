#! Set hparams in configs/autoencoder_module/vae.yaml, or below:
dataset=perov_5  #   mp_20 / perov_5 / carbon_24 / mpts_52
latent_dim=8       # 4 / 6 / 8
loss_weight_kl=0.0001  # 0.0001 / 0.00001

#! (for logging purposes)
latent_str="latent@${latent_dim}"
kl_str="kl@${loss_weight_kl}"
name="${dataset}_cond_vae_${latent_str}_${kl_str}"

application="python src/train_autoencoder.py"

options="trainer=gpu logger=wandb name=$name \
    data=$dataset \
    ++autoencoder_module.latent_dim=$latent_dim \
    ++autoencoder_module.loss_weight_kl=$loss_weight_kl"

CMD="HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 $application $options"

echo "Running command:"
echo "$CMD"
echo

eval $CMD