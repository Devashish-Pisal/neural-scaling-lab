

COLUMN_NAMES = {
    "model_name": "Model",
    "patch_size": "Patch Size",
    "ds_name": "Dataset Name",
    "fg_range": "FG Range",
    "bg_range": "BG Range",
    "ds_size_fraction": "Dataset Fraction",
    "calc_ds_samples": "Calculated DS Samples",
    "calc_steps": "Calculated Steps",
    "calc_epochs": "Epochs",
    "real_ds_samples": "Real DS Samples",
    "real_steps": "Real Steps",
    "real_FLOPs": "FLOPs",
    "min_val_loss": "Val/Loss (Min)",
    "max_val_acc1": "Val/Acc1 (Max)",
    "max_val_acc5": "Val/Acc5 (Max)",
    "min_train_loss": "Train/Loss (Min)",
}


WANDB_RUN_CONFIG = {
    "vit-s/16": {
        "imagenet_run_ids": ["7zhrps5n", "f89h2iep", "q7pwnnmk", "p4lmf22i"], # for ds size [1.0, 0.5, 0.25, 0.10]
        "fornet_run_ids": {
            1.0 : [],
            0.5 : [],
            0.25 : [],
            0.10 : [],
        },
    },
    # add vit-s/32, vit-b/16, vit-s/32, vit-b/28, etc.
}


EXPERIMENT_CONSTANTS = {
    "global_batch_size": 2048,
    "model_parameters":{
        "vit-s-16": 22_000_000,
        "vit-s-32": 0,
        "vit-b-16": 0,
    }
}