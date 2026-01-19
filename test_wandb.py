import wandb

wandb.init(project="ADiT")
wandb.log({"test_metric": 1})
wandb.finish()
