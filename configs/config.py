
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


# ============================================================================
# STATUS: Rewritten 2026-08-25 against Chapter 4 ("Experiment Design") of
# 2026-08-25_thesis_third-draft.txt, which the user has designated as the
# source of truth, superseding 02_EXPERIMENTAL_DESIGN.txt /
# 03_TRAINING_SCALING_AND_CONFIG.txt wherever they conflict.
#
# KEY CHANGE vs. the prior (internal-notes-driven) version of this file:
# the 100-/200-epoch baselines are NOT restricted to patch-16 models
# (ViT-S/16, ViT-B/16, ViT-L/16) as 03_TRAINING_SCALING_AND_CONFIG.txt /
# 02_EXPERIMENTAL_DESIGN.txt (Group 4) stated. Thesis Table 4.4
# ("Scaling Law Analysis") explicitly runs ALL 8 models across all 3
# baselines: "8 models x 2 datasets x 4 subsets x 3 training budgets = 192"
# - this arithmetic is only consistent with all 8 models x all 3 baselines,
# and the user explicitly confirmed this reading. The previous
# patch-16-only restriction is DISCARDED.
# ============================================================================

# Compute-aware epoch schedule per baseline (S ∝ D^0.6, see
# 03_TRAINING_SCALING_AND_CONFIG.txt and thesis Table 4.4's right-hand
# epoch/step table). The 300-epoch row is VERIFIED against
# 20260804_MAINEXPERIMENTRUNS.csv and matches the thesis's "Baseline
# epochs...186k" row exactly. The 100- and 200-epoch rows are DERIVED
# (NOT YET RUN) - re-verify against your actual scheduler/config output
# before trusting them to validate real runs.
BASELINE_EPOCH_SCHEDULES = {
    300: {"0-10": 754, "0-25": 522, "0-50": 396, "0-100": 300},   # VERIFIED
    200: {"0-10": 502, "0-25": 348, "0-50": 264, "0-100": 200},   # DERIVED
    100: {"0-10": 251, "0-25": 174, "0-50": 132, "0-100": 100},   # DERIVED
}


# The 5 models that get the FULL 4x4 fg x bg ForNet grid (Foreground-
# Background Contribution Analysis, thesis Table 4.5), and ONLY at the
# 300-epoch baseline. Thesis: "Tier-1 Models Ti/16, S/16, S/32, B/16, B/28
# ... Baseline epochs 300". All 8 models (this set plus B/32, L/16, L/32)
# still get the Scaling Law Analysis (ImageNet + ForNet-cos-bg100%) at all
# 3 baselines - this set only controls which models ALSO get the extra
# bg_range=10/25/50 columns at baseline=300.
FG_BG_GRID_MODELS = ("vit-ti/16", "vit-s/16", "vit-s/32", "vit-b/16", "vit-b/28")


# Models in-scope for the Background Subset Size Ablation (thesis Table
# 4.6 / formerly "Group 3" extreme-BG ablation). ViT-Ti/16 is NOT part of
# this - confirmed independently by BOTH 02_EXPERIMENTAL_DESIGN.txt
# (Decision Log 2026-08-16, item 5) AND thesis Table 4.6 (only lists S/16,
# B/16, L/16). This double-confirms that 04_OUTPUTS.txt's mention of
# "expanding to Ti/16, B/16, L/16" is a wording error in that file - fix
# it there, not here.
EXTREME_BG_INSCOPE_MODELS = ("vit-s/16", "vit-b/16", "vit-l/16")


# Tier 1 (5 models, VERIFIED at 300 epochs) + Tier 2 (3 models, PLANNED) as
# named in thesis Table 4.2. Restructured (2026-08-25) so every model has a
# sub-dict keyed by baseline_epochs (300/200/100), per Chapter 4's Scaling
# Law Analysis spanning all 3 baselines for all 8 models.
#
# Shape per (model, baseline_epochs):
#   "imagenet_run_ids": {1.00: [id], 0.50: [id], 0.25: [id], 0.10: [id]}
#   "fornet_run_ids":   {1.00: [...], 0.50: [...], 0.25: [...], 0.10: [...]}
#
# For fornet_run_ids, the list length signals which analysis a row belongs
# to and is validated accordingly by build_main_experiment_runs_database.py:
#   - length 4 (ids in descending bg_range order ["0-100","0-50","0-25","0-10"])
#     -> the full fg x bg grid (Foreground-Background Contribution Analysis,
#     Table 4.5). ONLY valid when baseline_epochs == 300 AND model is in
#     FG_BG_GRID_MODELS - the builder hard-fails otherwise, to catch
#     config typos.
#   - length 1 (bg_range fixed at "0-100")
#     -> Scaling Law Analysis only (Table 4.4). Valid for any model at any
#     baseline, and it's the ONLY valid shape for baseline in (100, 200),
#     and for the 3 Tier-2 models at baseline 300 too.
#
# Leave a model's baseline entry as {} (both sub-dicts empty) until its
# runs finish - the builder skips empty entries entirely rather than
# raising, so this file always reflects the FULL intended 192-run design
# even while most cells are still empty.
MAIN_EXPERIMENT_ID_MAPPING = {
    "vit-s/16": {
        # ---- VERIFIED against 20260804_MAINEXPERIMENTRUNS.csv ----
        300: {
            "imagenet_run_ids": {
                1.00: ["7zhrps5n"],
                0.50: ["f89h2iep"],
                0.25: ["q7pwnnmk"],
                0.10: ["p4lmf22i"],
            },
            "fornet_run_ids": {
                # list order: ["0-100", "0-50", "0-25", "0-10"]
                1.00: ["bdz7rm89", "37k5y5wt", "g50zkmfl", "o5jsrx8z"],
                0.50: ["ztp1d4go", "tad9m3tt", "9dnrukwp", "aggfcwxo"],
                0.25: ["41kt8ijm", "e83apfnv", "5ndq2oce", "w35da4yn"],
                0.10: ["i8w015s3", "gsrrhci9", "nshojpql", "giqxrx3p"],
            },
        },
        200: {
            "imagenet_run_ids": {
                1.00:["zz3lvgdj"],
                0.50:["filbb6ys"],
                0.25:["327zz6r9"],
                0.10:["slqs1sd9"],
            },
            "fornet_run_ids": { # BG range is fixed to 0-100
                1.00: ["heutcy1f"],
                0.50: ["nuhjmrn8"],
                0.25: ["nb5lryla"],
                0.10: ["kfl3hd12"],
            }
        },
        100: {
            "imagenet_run_ids": {
                1.00:["4mohggw5"],
                0.50:["n3qqem0x"],
                0.25:["cpnguebb"],
                0.10:["r3whglmv"],
            },
            "fornet_run_ids": {
                1.00:["x7fr6c2s"],
                0.50:["4mv6c6s3"],
                0.25:["mxvb107b"],
                0.10:["v3wir9c9"],
            }
        },
    },
    "vit-s/32": {
        300: {
            "imagenet_run_ids": {
                1.00: ["sefj86tp"],
                0.50: ["rvpvlg79"],
                0.25: ["x2vp97f9"],
                0.10: ["znjol1kh"],
            },
            "fornet_run_ids": {
                1.00: ["6kca8s1e", "8a37yvll", "9ex8msy3", "23c0dskh"],
                0.50: ["28seu0s6", "m6z2z4zh", "8wbczxzc", "zsyslyse"],
                0.25: ["gvphs0tz", "kx64opuq", "j2m7ubf7", "6i20wdfk"],
                0.10: ["c9qiljuf", "uc1lodox", "6m8uo7ep", "llnetkcv"],
            },
        },
        200: {
            "imagenet_run_ids": {
        #        1.00: [""],
        #        0.50: [""],
        #        0.25: [""],
        #        0.10: [""],
            },
            "fornet_run_ids": {
        #        1.00: [""],
        #        0.50: [""],
        #        0.25: [""],
        #        0.10: [""],
            }
        },
        100: {
            "imagenet_run_ids": {
        #        1.00: [""],
        #        0.50: [""],
        #        0.25: [""],
        #        0.10: [""],
            },
            "fornet_run_ids": {
        #        1.00: [""],
        #        0.50: [""],
        #        0.25: [""],
        #        0.10: [""],
            }
        },
    },
    "vit-ti/16": {
        300: {
            "imagenet_run_ids": {
                1.00: ["44tt3jl8"],
                0.50: ["4nnn02jg"],
                0.25: ["5w0fhgr1"],
                0.10: ["yax881w3"],
            },
            "fornet_run_ids": {
                1.00: ["3a1hoa23", "u71lfrpq", "nl3mznoe", "xkdspyiu"],
                0.50: ["x727q5mc", "a0wp34ot", "tolyodn8", "issyug6e"],
                0.25: ["fty4m1bk", "4amdgv5z", "96pst16d", "e6koz253"],
                0.10: ["jxhfqu0o", "4jcnfqct", "mxo04qqe", "66xvvxq5"],
            },
        },
        200: {
            "imagenet_run_ids": {
                1.00: ["oawas7qz"],
                0.50: ["4nx1iwsc"],
                0.25: ["2avlhuv2"],
                0.10: ["xq0v0ijl"],
            },
            "fornet_run_ids": {
                1.00: ["mu3posou"],
                0.50: ["9vdbxhl6"],
                0.25: ["8vm4a6bw"],
                0.10: ["agifnz49"],
            }
        },
        100: {
            "imagenet_run_ids": {
                1.00: ["efrr7ilp"],
                0.50: ["71566g7f"],
                0.25: ["o2u47ozp"],
                0.10: ["mx4fqsly"],
            },
            "fornet_run_ids": {
                1.00: ["z34eorvm"],
                0.50: ["ge27uy0u"],
                0.25: ["5kzp7t8l"],
                0.10: ["o1vgpsq2"],
            }
        },
    },
    "vit-b/16": {
        300: {
            "imagenet_run_ids": {
                1.00: ["clm520y2"],
                0.50: ["nsbmttiq"],
                0.25: ["g5k2agtt"],
                0.10: ["pksgtthn"],
            },
            "fornet_run_ids": {
                1.00: ["v9l4gigx", "02nd637f", "m1ikhd20", "gtm0m4sa"],
                0.50: ["so20x9ri", "dlvjqdb3", "pnx3dj8y", "4aq2b6yp"],
                0.25: ["978sjuch", "ykg1mkgu", "fj3zrxk3", "niyyfist"],
                0.10: ["dug2gz7a", "pagtn2eb", "0jxr740c", "e7kasssf"],
            },
        },
        200: {
            "imagenet_run_ids": {
                1.00:["tuypq3qp"],
                0.50:["v2fcpxho"],
                0.25:["h3tonrul"],
                0.10:["kq86k0tj"],
            },
            "fornet_run_ids": { # BG range is fixed to 0-100
                1.00: ["0d9hx2jy"],
                0.50: ["hqv90fr6"],
                0.25: ["t05n4m6r"],
                0.10: ["ppv4p8j9"],
            }
        },
        100: {
            "imagenet_run_ids": {
                1.00:["i9kj2uws"],
                0.50:["zhntvp28"],
                0.25:["1w7paazi"],
                0.10:["qbi5i9ol"],
            },
            "fornet_run_ids": {
                1.00: ["ri85gdz5"],
                0.50: ["4gh9bpyc"],
                0.25: ["2qhbbxbk"],
                0.10: ["285p09ge"],
            }
        },
    },
    "vit-b/28": {
        300: {
            "imagenet_run_ids": {
                1.00: ["fycsyzq9"],
                0.50: ["ldwp7rc4"],
                0.25: ["s1qz47e7"],
                0.10: ["i20ibmvl"],
            },
            "fornet_run_ids": {
                1.00: ["j78h3pfz", "qur9r6cq", "ibq5gjii", "raaq0z3t"],
                0.50: ["06y5h4o8", "yapcwhst", "qnmk8dxg", "cplyjt5a"],
                0.25: ["p9ywochc", "a5k12id0", "4s81jg7t", "6u8aa0o4"],
                0.10: ["fs5cjzbo", "8rolw2pg", "adwxtmb5", "s54kp6og"],
            },
        },
        200: {"imagenet_run_ids": {}, "fornet_run_ids": {}},
        100: {"imagenet_run_ids": {}, "fornet_run_ids": {}},
    },

    # ---- Tier 2 (thesis Table 4.2): Scaling Law Analysis ONLY (Table 4.4)
    # - no full fg x bg grid for these (they are NOT in FG_BG_GRID_MODELS,
    # per thesis Table 4.5 restricting that analysis to the 5 models above).
    # ForNet runs are fixed at bg_range="0-100" at every baseline. All
    # PLANNED, NOT YET RUN as of 2026-08-25 - fill in as they complete.
    "vit-b/32": {
        300: {
            "imagenet_run_ids": {
                1.00:["1yhj16m2"],
                0.50:["ewdz4kti"],
                0.25:["qwpe8cyl"],
                0.10:["gmnoz8lt"],
            },
            "fornet_run_ids": {
                # bg_range="0-100"
                1.00: ["hnmbuj0v"],
                0.50: ["17y0d1ny"],
                0.25: ["ii3xu9d6"],
                0.10: ["ppf5fxz3"],
            }
        },
        200: {"imagenet_run_ids": {}, "fornet_run_ids": {}},
        100: {"imagenet_run_ids": {}, "fornet_run_ids": {}},
    },
    "vit-l/16": {
        300: {
            "imagenet_run_ids": {
                1.00:["pgwxjq1c"],
                0.50:["7nu9o928"],
                0.25:["zasuq0q7"],
                0.10:["vo052ypw"],
            },
            "fornet_run_ids": {
                # bg_range="0-100"
                1.00: ["qleelnaz"],
                0.50: ["oor2s32c"],
                0.25: ["ubpmv79o"],
                0.10: ["6yj73t8h"],
            }
        },
        200: {
            "imagenet_run_ids": {
#                1.00:["i3bkfx2k"], # change ID here
#                0.50:["dh4e1tui"],
#                0.25:["bvmqkxc2"],
#                0.10:["p0t76sn1"],
            },
            "fornet_run_ids": {
                # bg_range="0-100"
#                1.00: ["fgu11hps"],
#                0.50: ["3uqpz93d"],
#                0.25: ["ovso5kd3"],
#                0.10: ["hqrdwed9"],
            }
        },
        100: {"imagenet_run_ids": {}, "fornet_run_ids": {}},
    },
    "vit-l/32": {
        300: {
            "imagenet_run_ids": {
                1.00: ["o2zko39l"],
                0.50: ["aegl1rcc"],
                0.25: ["7vvk8e6s"],
                0.10: ["djdb0fh1"],
            },
            "fornet_run_ids": {
                # bg_range="0-100"
                1.00: ["tr8n4gwq"],
                0.50: ["7rogo0eq"],
                0.25: ["4gmc7516"],
                0.10: ["fbywzca6"],
            }
        },
        200: {"imagenet_run_ids": {}, "fornet_run_ids": {}},
        100: {"imagenet_run_ids": {}, "fornet_run_ids": {}},
    },
}


# ============================================================================
# Background Subset Size Ablation (thesis Table 4.6, formerly "Group 3").
# fg_range is FIXED at "0-100" for every entry here (thesis: "we fix the
# foreground subset size to 100%"). Table 4.6 does not restate a baseline
# epoch count for this ablation - ASSUMPTION (not explicitly stated in
# Chapter 4, carried over from the pre-existing pilot data's validation
# logic, which only ever accepted the 300-epoch schedule): baseline_epochs
# = 300 throughout. Flag/correct this if a different baseline is intended.
#
# Table 4.6's exact grid:
#   S/16, oip=0.0: all 7 points   | S/16, oip=cos: all 7 points
#   B/16, oip=0.0: all 7 points   | B/16, oip=cos: only bg=10%, bg=100%
#   L/16, oip=0.0: all 7 points   | L/16, oip=cos: only bg=10%, bg=100%
#
# IMPORTANT: for B/16 (and one of L/16's two) oip=cos points, thesis
# Chapter 4 says these are the SAME runs used elsewhere ("some runs for
# the cosine mixing strategy are skipped [i.e. not newly run]"):
#   - B/16 oip=cos, bg=10% and bg=100%, fg=100%, 300 epochs: this is
#     IDENTICAL to MAIN_EXPERIMENT_ID_MAPPING["vit-b/16"][300]["fornet_run_ids"][1.00],
#     entries index 0 (bg=100%, id "v9l4gigx") and index 3 (bg=10%, id "gtm0m4sa").
#   - L/16 oip=cos, bg=100%, fg=100%, 300 epochs: will be IDENTICAL to
#     MAIN_EXPERIMENT_ID_MAPPING["vit-l/16"][300]["fornet_run_ids"][1.00][0]
#     once that Scaling Law Analysis run exists.
# ============================================================================
EXTREME_BG_ID_MAPPING = {
    "oip-cos": {
        "vit-s/16": {
            1.00: {  # FG Percentage: 100%; FG Range: "0-100"
                1.0:"bdz7rm89",         # 100%
                0.1:"o5jsrx8z" ,         # 10%
                # 0.05: "a6n0flf1",      # 5%
                0.01: "2lpp8zmr",      # 1%
                0.001: "jvgppsf2",     # 0.1%
                # 0.0005: "cmz99zd5",    # 0.05%
                0.0001: "wz7pd7uj",    # 0.01%
                0.00001: "yr8ifqbl",   # 0.001%
                0.000001: "5crvkryy",  # 0.0001%
            }
        },
        "vit-b/16":{
            1.00: {
                1.0: "v9l4gigx",    # 100%
                0.1: "gtm0m4sa",    # 10%
            }
        },
        "vit-l/16": {
            1.00: {
                1.0:"qleelnaz",         # 100%
                # 0.1:"" ,        # 10%
            }
        },
    },
    "oip-0.0": {
        "vit-s/16": {
            1.00: { # (bg_range "0-100")
                1.0: "iozew7nb",       # 100%
                0.1: "84r06ykk",       # 10%
                0.01: "bxk80blj",      # 1%
                0.001: "toh8jtig",     # 0.1%
                0.0001: "6xcu727n",    # 0.01%
                0.00001: "wwdnerzd",   # 0.001%
                0.000001: "p2rs2wqi",  # 0.0001%
            }
        },
        "vit-b/16": {
            1.00: {
                1.0:      "zcepwhr5",  # 100%
                0.1:      "flwmtoe4",  # 10%
                0.01:     "fvgxf3zh",  # 1%
                0.001:    "fs68fmrs",  # 0.1%
                0.0001:   "a2z5yw5c",  # 0.01%
                0.00001:  "kahf04gg",  # 0.001%
                0.000001: "zbz6tfnx",  # 0.0001%
            }
        },
        "vit-l/16": {
            1.00: {
                1.0:      "oznu5eam",  # 100%
                0.1:      "vbpq529e",  # 10%
                0.01:     "ehm772an",  # 1%
                0.001:    "rgbg93ao",  # 0.1%
                0.0001:   "ee6cmsc4",  # 0.01%
                0.00001:  "313lwfuo",  # 0.001%
                0.000001: "8gosxiqz",  # 0.0001%
            }
        },
    }
}


EXPERIMENT_CONSTANTS = {
    "dataset_image_resolution": 224, # (224 x 224) px image resolution of ImageNet-1k dataset samples
    "global_batch_size": 2048,
    "model_parameters":{
        # NOTE: only the 5 completed Tier-1 models are populated below.
        # vit-b/32, vit-l/16, vit-l/32 are MISSING - configs/metric.py's
        # METRICS dict is also missing the same three models. Both must be
        # populated with MEASURED values (per 04_OUTPUTS.txt's own
        # convention: "computed programmatically... using the top
        # checkpoint") before running any build script against runs for
        # these models - do not fabricate these numbers.
        "vit-ti/16": 5_721_832,
        "vit-s/16": 22_059_496,
        "vit-s/32": 22_887_784,
        "vit-b/16": 86_585_320,
        "vit-b/28": 87_700_456,
        "vit-b/32": 88_241_896,
        "vit-l/16": 304_374_760,
        "vit-l/32": 306_583_528,
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
    "train_dataset_fraction": "Dataset fraction", # alternative name fg_fraction
    "bg_fraction": "Background subset fraction",
    "total_epochs": "Epochs",
    "baseline_epochs": "Baseline epoch budget",  # NEW (2026-08-25): which of
    # the 3 Scaling Law Analysis baselines (100/200/300, see
    # BASELINE_EPOCH_SCHEDULES) this run belongs to. Always populated
    # explicitly via save_run_data_to_db's extra_fields - never left as
    # the raw DB_COLUMNS default label.

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
    "parameter_count": "Parameter count",
    "macs_per_image": "MACs per image",
    "flops_per_image": "FLOPs per image",
    "total_flops": "FLOPs",
    "crossover_epoch_val_loss": "Crossover val/loss epoch",
    "crossover_epoch_val_acc1": "Crossover val/acc1 epoch",
    "crossover_epoch_val_acc5": "Crossover val/acc5 epoch",
    "total_runtime": "Runtime (in sec.)",
    "gpu_partition": "GPU Partition"

    # Total 28 columns
}


# COUNT is taken before pruning (after pruning, count reduces little bit)
FG_RANGE_COUNT_MAPPING = {
    "0-10": 127_456,
    "0-25": 318_639,
    "0-50": 637_279,
    "0-100": 1_274_557
}


# COUNT is taken before pruning (after pruning, count reduces little bit)
# Every bg_range code needed by both the extreme-BG ablation grid and the
# pre-existing ViT-S/16 oip=cos pilot grid is already present below.
BG_RANGE_COUNT_MAPPING = {
    "0-0.0001": 1,
    "0-0.001": 13,
    "0-0.01": 127,
    # "0-0.05": 637,
    "0-0.1": 1_275,
    "0-1": 12_746,
    # "0-5": 63_782,
    "0-10": 127_456,
    "0-25": 318_639,
    "0-50": 637_279,
    "0-100": 1_274_557,
    "null": None
}