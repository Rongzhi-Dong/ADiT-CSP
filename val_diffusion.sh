#!/usr/bin/env bash
# Local evaluation script for DiT diffusion model
# - No wandb
# - No env activation
# - No terminal arguments
# - Edit checkpoint paths in THIS file only

# set -eu -o pipefail

# ============================================================
#               EDIT THESE PATHS LATER
# ============================================================

# Path to DiT (diffusion) checkpoint
# DIFF_CKPT="/home/rongzhi/ADiT-CSP/logs/train_diffusion/runs/DiT-B_vae_latent@9_kl@0.00001_2026-01-25_17-12-41/checkpoints/ldm-epoch@1499-step@159000-val_mp20_valid_rate@0.2120.ckpt"
DIFF_CKPT="/home/rongzhi/ADiT-CSP/pre-ckpt/ldm.ckpt"

# Path to VAE checkpoint
# VAE_CKPT="/home/rongzhi/ADiT-CSP/logs/train_autoencoder/runs/vae_latent@8_kl@0.00001_2026-01-19_23-43-03/checkpoints/vae-epoch@3049-step@323300-val_mp20_match_rate@0.8126.ckpt"
VAE_CKPT="/home/rongzhi/ADiT-CSP/pre-ckpt/vae.ckpt"


name="test_pre-ckpt"


# d_x=9 ##(D+1)
# kl=0.00001
# name="val_DiT-S_vae_latent@${d_x}_kl@${kl}"


application="python src/eval_diffusion.py"
# workdir="./"

# options="trainer=gpu logger=wandb name=$name \
#     ++diffusion_module.denoiser.d_x=$d_x"

options="trainer=gpu logger=wandb name=$name"

CMD="HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 $application $options"

# Go to working directory


echo "Running command:"
echo "$CMD"
echo

# Execute
eval $CMD

# # ============================================================
# #               OPTIONAL OVERRIDES
# # ============================================================

# # Add any extra Hydra overrides here if needed
# # Example:
# # EXTRA_OVERRIDES="trainer=cpu diffusion_module.batch_size=8"
# EXTRA_OVERRIDES=""

# # ============================================================
# #               INTERNALS (DON'T TOUCH)
# # ============================================================

# APPLICATION="python src/eval_diffusion.py"
# WORKDIR="."
# LOGGER_OVERRIDE="logger=null"

# cd "${WORKDIR}"

# echo "Starting local evaluation"
# echo

# # Build Hydra overrides
# OVERRIDES=""

# # Diffusion checkpoint (required)
# if [ "${DIFF_CKPT}" != "null" ]; then
#   OVERRIDES+=" ckpt_path='${DIFF_CKPT}'"
# else
#   echo "WARNING: DIFF_CKPT is null (edit eval_local.sh to set it)"
# fi

# # VAE checkpoint (optional but recommended)
# if [ "${VAE_CKPT}" != "null" ]; then
#   OVERRIDES+=" diffusion_module.vae_ckpt='${VAE_CKPT}'"
# else
#   echo "WARNING: VAE_CKPT is null (edit eval_local.sh to set it)"
# fi

# # Disable wandb unless explicitly overridden
# OVERRIDES+=" ${LOGGER_OVERRIDE}"

# # Append extra user overrides
# if [ -n "${EXTRA_OVERRIDES}" ]; then
#   OVERRIDES+=" ${EXTRA_OVERRIDES}"
# fi

# CMD="HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 ${APPLICATION} ${OVERRIDES}"

# echo
# echo "Running command:"
# echo
# echo "${CMD}"
# echo

# eval ${CMD}




# # Whether to send logs to Weights & Biases
# USE_WANDB=true         # set to true to enable W&B logging

# # W&B settings (only used if USE_WANDB=true)
# WANDB_PROJECT="eval_diffusion"   # project name in W&B
# WANDB_ENTITY=""                  # your W&B entity/team (leave empty if not using)
# WANDB_RUN_NAME=""                # custom run name; empty -> auto

# # WANDB mode: "online" (default), "offline" (store locally), or "dryrun"
# # If you want to avoid contacting W&B, set WANDB_MODE=dryrun
# WANDB_MODE="online"

# # Optional: directory for wandb local files (if desired)
# WANDB_DIR=""

# # Extra Hydra overrides you might want to add
# EXTRA_OVERRIDES=""

# # =========================
# # INTERNALS - do not edit below
# # =========================

# APPLICATION="python src/eval_diffusion.py"
# WORKDIR="."
# cd "${WORKDIR}"
# echo "Working dir: $(pwd)"
# echo

# # Basic checks
# if [ "${DIFF_CKPT}" = "null" ]; then
#   echo "ERROR: DIFF_CKPT is null — edit the script and set DIFF_CKPT to your diffusion ckpt path."
#   exit 1
# fi
# if [ "${VAE_CKPT}" = "null" ]; then
#   echo "WARNING: VAE_CKPT is null — if your eval requires a VAE checkpoint, set VAE_CKPT in the script."
# fi

# # Build Hydra overrides
# OVERRIDES="ckpt_path='${DIFF_CKPT}'"
# if [ "${VAE_CKPT}" != "null" ]; then
#   OVERRIDES+=" diffusion_module.vae_ckpt='${VAE_CKPT}'"
# fi

# # Configure W&B if requested
# if [ "${USE_WANDB}" = "true" ]; then
#   # Expose WANDB env vars for the process
#   export WANDB_MODE="${WANDB_MODE}"
#   if [ -n "${WANDB_PROJECT}" ]; then
#     export WANDB_PROJECT="${WANDB_PROJECT}"
#   fi
#   if [ -n "${WANDB_ENTITY}" ]; then
#     export WANDB_ENTITY="${WANDB_ENTITY}"
#   fi
#   if [ -n "${WANDB_DIR}" ]; then
#     export WANDB_DIR="${WANDB_DIR}"
#   fi

#   # Compose Hydra wandb logger overrides.
#   # The exact keys may differ depending on your Hydra + Lightning logger implementation,
#   # but `logger=wandb` and `logger.wandb.*` are common patterns.
#   # If your code uses a different key, edit these lines accordingly.
#   WAND_B_OVERRIDES="logger=wandb"
#   if [ -n "${WANDB_PROJECT}" ]; then
#     WAND_B_OVERRIDES+=" logger.wandb.project='${WANDB_PROJECT}'"
#   fi
#   if [ -n "${WANDB_ENTITY}" ]; then
#     WAND_B_OVERRIDES+=" logger.wandb.entity='${WANDB_ENTITY}'"
#   fi
#   if [ -n "${WANDB_RUN_NAME}" ]; then
#     WAND_B_OVERRIDES+=" logger.wandb.name='${WANDB_RUN_NAME}'"
#   fi

#   OVERRIDES+=" ${WAND_B_OVERRIDES}"
#   echo "W&B logging ENABLED (WANDB_MODE=${WANDB_MODE})"
#   echo "  WANDB_PROJECT=${WANDB_PROJECT}"
#   echo "  WANDB_ENTITY=${WANDB_ENTITY}"
#   if [ -n "${WANDB_RUN_NAME}" ]; then
#     echo "  WANDB_RUN_NAME=${WANDB_RUN_NAME}"
#   fi
# else
#   # disable wandb via Hydra logger override
#   OVERRIDES+=" logger=null"
#   echo "W&B logging DISABLED (logger=null)."
# fi

# # Append any user-provided extra overrides
# if [ -n "${EXTRA_OVERRIDES}" ]; then
#   OVERRIDES+=" ${EXTRA_OVERRIDES}"
# fi

# # Helpful reminder: user must be logged in or have WANDB_API_KEY if using online mode
# if [ "${USE_WANDB}" = "true" ] && [ "${WANDB_MODE}" = "online" ]; then
#   if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_ENTITY:-}" ]; then
#     echo
#     echo "NOTE: You should be logged into wandb (run 'wandb login') or set WANDB_API_KEY in env."
#     echo "If you prefer not to connect, set WANDB_MODE='dryrun' or USE_WANDB=false."
#     echo
#   fi
# fi

# d_x=9
# options= "diffusion_module.denoiser.d_x=$d_x"

# CMD="HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 ${APPLICATION} ${OVERRIDES} ${options}"

# echo
# echo "About to run:"
# echo
# echo "${CMD}"
# echo

# # Execute (user runs script in appropriate env)
# eval ${CMD}

