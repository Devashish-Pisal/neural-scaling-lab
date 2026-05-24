

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
            "1.0" : [],
            "0.5" : [],
            "0.25" : [],
            "0.10" : [],
        },
    },
    # add vit-s/32, vit-b/16, vit-s/32, vit-b/28, etc.
}


EXPERIMENT_CONSTANTS = {
    "global_batch_size": 2048,
    "model_parameters":{
        "vit-s/16": 22_000_000,
        "vit-s/32": 0,
        "vit-b/16": 0,
    }
}



DB_COLUMNS = {
    # Inputs
    "run_id": "ID",
    "run_name": "Name",
    "model_name": "Model",
    "train_dataset_name": "Training dataset",
    "fg_range": "FG range",
    "bg_range": "BG range",
    "fg_count": "Foregrounds (#)",
    "bg_count": "Backgrounds (#)",
    "train_dataset_size": "Dataset size (#)",
    "train_dataset_fraction": "Dataset fraction",
    "total_epochs": "Epochs",

    # Outputs
    "min_train_loss": "Train/Loss (min)",
    "min_train_loss_epoch": "Min. train/loss epoch",
    "min_val_loss": "Val/Loss (min)",
    "min_val_loss_epoch": "Min. val/loss epoch",
    "max_val_acc1": "Val/Acc1 (max)",
    "max_val_acc1_epoch": "Max. val/acc1 epoch",
    "max_val_acc5": "Val/Acc5 (max)",
    "max_val_acc5_epoch": "Max. val/acc5 epoch",
    "final_val_loss": "Final val/loss",
    "final_val_acc1": "Final val/acc1",
    "final_val_acc5": "Final val/acc5",
    "final_train_loss": "Final train/loss",
    "total_runtime": "Runtime (in sec.)",
    "steps_per_epoch": "Steps per epoch",
    "total_steps": "Total steps",
    "total_flops": "FLOPs",
    "gpu_partition": "GPU Partition"

    # Total 27 columns
}



FG_RANGE_COUNT_MAPPING = {
    "0-10": 127_511,
    "0-25": 318_624,
    "0-50": 637_328,
    "0-100": 1_274_557
}

BG_RANGE_COUNT_MAPPING = {
    "0-10": 113_767,
    "0-25": 284_111,
    "0-50": 569_433,
    "0-100": 1_145_497,
    "null": None
}