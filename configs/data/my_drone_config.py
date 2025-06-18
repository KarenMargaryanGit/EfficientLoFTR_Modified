from configs.data.base import cfg

cfg.DATASET.TRAINVAL_DATA_SOURCE = "Custom"
cfg.DATASET.TRAIN_DATA_ROOT = "train_dataset_sample_copy/images"
cfg.DATASET.TRAIN_NPZ_ROOT = "train_dataset_sample_copy/meta"
cfg.DATASET.TRAIN_LIST_PATH = "train_dataset_sample_copy/pair_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

# For validation (if you have validation pairs)
cfg.DATASET.VAL_DATA_ROOT = "train_dataset_sample_copy/images"
cfg.DATASET.VAL_NPZ_ROOT = "train_dataset_sample_copy/meta"
cfg.DATASET.VAL_LIST_PATH = "train_dataset_sample_copy/pair_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0