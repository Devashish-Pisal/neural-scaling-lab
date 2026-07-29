

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
        "imagenet_run_ids": {
            1.00: ["7zhrps5n"],
            0.50: ["f89h2iep"],
            0.25: ["q7pwnnmk"],
            0.10: ["p4lmf22i"],
        },
        "fornet_run_ids": {
            # key : dataset fraction (fg_range)
            # value: list of ids with different bg_range in descending order ["0-100", "0-50", "0-25", "0-10"]
            1.00 : ["bdz7rm89", "37k5y5wt", "g50zkmfl", "o5jsrx8z"],
            0.50 : ["ztp1d4go", "tad9m3tt", "9dnrukwp", "aggfcwxo"],
            0.25 : ["41kt8ijm", "e83apfnv", "5ndq2oce", "w35da4yn"],
            0.10 : ["i8w015s3", "gsrrhci9", "nshojpql", "giqxrx3p"],
        },
    },
    "vit-s/32": {
        "imagenet_run_ids": {
            1.00: ["sefj86tp"],
            0.50: ["rvpvlg79"],
            0.25: ["x2vp97f9"],
            0.10: ["znjol1kh"],
        },
        "fornet_run_ids": {
            # key : dataset fraction (fg_range)
            # value: list of ids with different bg_range in descending order ["0-100", "0-50", "0-25", "0-10"]
            1.00: ["6kca8s1e", "8a37yvll", "9ex8msy3", "23c0dskh"],
            0.50: ["28seu0s6", "m6z2z4zh", "8wbczxzc", "zsyslyse"],
            0.25: ["gvphs0tz", "kx64opuq", "j2m7ubf7", "6i20wdfk"],
            0.10: ["c9qiljuf", "uc1lodox", "6m8uo7ep", "llnetkcv"],
        },
    },
    "vit-ti/16": {
        "imagenet_run_ids": {
            1.00: ["44tt3jl8"],
            0.50: ["4nnn02jg"],
            0.25: ["5w0fhgr1"],
            0.10: ["yax881w3"],
        },
        "fornet_run_ids": {
            # key : dataset fraction (fg_range)
            # value: list of ids with different bg_range in descending order ["0-100", "0-50", "0-25", "0-10"]
            1.00: ["3a1hoa23", "u71lfrpq", "nl3mznoe", "xkdspyiu"],
            0.50: ["x727q5mc", "a0wp34ot", "tolyodn8", "issyug6e"],
            0.25: ["fty4m1bk", "4amdgv5z", "96pst16d", "e6koz253"],
            0.10: ["jxhfqu0o", "4jcnfqct", "mxo04qqe", "66xvvxq5"],
        },
    },
    "vit-b/16": {
        "imagenet_run_ids": {
            1.00: ["clm520y2"],
            0.50: ["nsbmttiq"],
            0.25: ["g5k2agtt"],
            0.10: ["pksgtthn"],
        },
        "fornet_run_ids": {
            # key : dataset fraction (fg_range)
            # value: list of ids with different bg_range in descending order ["0-100", "0-50", "0-25", "0-10"]
            1.00: ["v9l4gigx", "02nd637f", "m1ikhd20", "gtm0m4sa"],
            0.50: ["so20x9ri", "dlvjqdb3", "pnx3dj8y", "4aq2b6yp"],
            0.25: ["978sjuch", "ykg1mkgu", "fj3zrxk3", "niyyfist"],
            0.10: ["dug2gz7a", "pagtn2eb", "0jxr740c", "e7kasssf"],
        },
    },
    "vit-b/28": {
        "imagenet_run_ids": {
            1.00: ["fycsyzq9"],
            0.50: ["ldwp7rc4"],
            0.25: ["s1qz47e7"],
            0.10: ["i20ibmvl"],
        },
        "fornet_run_ids": {
            # key : dataset fraction (fg_range)
            # value: list of ids with different bg_range in descending order ["0-100", "0-50", "0-25", "0-10"]
            1.00: ["j78h3pfz", "qur9r6cq", "ibq5gjii", "raaq0z3t"],
            0.50: ["06y5h4o8", "yapcwhst", "qnmk8dxg", "cplyjt5a"],
            0.25: ["p9ywochc", "a5k12id0", "4s81jg7t", "6u8aa0o4"],
            0.10: ["fs5cjzbo", "8rolw2pg", "adwxtmb5", "s54kp6og"],
        },
    },
}



EXPERIMENT_CONSTANTS = {
    "dataset_image_resolution": 224, # (224 x 224) px image resolution of ImageNet-1k dataset samples
    "global_batch_size": 2048,
    "model_parameters":{
        "vit-ti/16": 5_700_000,
        "vit-s/16": 22_000_000,
        "vit-s/32": 22_000_000,
        "vit-b/16": 86_000_000,
        "vit-b/28": 86_000_000,
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
    "steps_per_epoch": "Steps per epoch",
    "total_steps": "Total steps",
    "flops_per_epoch": "FLOPs per epoch",
    "total_flops": "FLOPs",
    "crossover_epoch_val_loss": "Crossover val/loss epoch",
    "crossover_epoch_val_acc1": "Crossover val/acc1 epoch",
    "crossover_epoch_val_acc5": "Crossover val/acc5 epoch",
    "total_runtime": "Runtime (in sec.)",
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