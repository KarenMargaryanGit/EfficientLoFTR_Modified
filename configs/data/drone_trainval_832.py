from configs.data.base import cfg

# ——— Drone dataset settings ———
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Drone"
cfg.DATASET.TRAIN_DATA_ROOT     = "data/images"             # folder with your raw drone & satellite images
cfg.DATASET.TRAIN_NPZ_ROOT      = "data/prepared_pairs"     # where prepare_drone_dataset.py wrote pair_XXXX.npz
cfg.DATASET.TRAIN_LIST_PATH     = "data/drone_pairs_list.txt"# one .npz basename per line, no “.npz”
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0                     # don’t filter out any of your few pairs

cfg.DATASET.TEST_DATA_SOURCE    = "Drone"
cfg.DATASET.VAL_DATA_ROOT       = cfg.DATASET.TRAIN_DATA_ROOT
cfg.DATASET.VAL_NPZ_ROOT        = cfg.DATASET.TRAIN_NPZ_ROOT
cfg.DATASET.VAL_LIST_PATH       = cfg.DATASET.TRAIN_LIST_PATH
cfg.DATASET.MIN_OVERLAP_SCORE_TEST  = 0.0

# Since you only have a handful of samples, you can keep this small:
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 10

# Keep the same resize as the original recipe (unless you want to change it)
cfg.DATASET.MGDPT_IMG_RESIZE = 832

# A short name for logging
cfg.DATASET.NPE_NAME = 'drone'
