from src.config.default import _CN as cfg

cfg.DATASET.TRAINVAL_DATA_SOURCE = "File"
cfg.DATASET.TRAIN_DATA_ROOT = "train_dataset_sample"
cfg.DATASET.TRAIN_NPZ_ROOT = "train_dataset_sample"
cfg.DATASET.TRAIN_LIST_PATH = "train_dataset_sample/drone_pairs_list.txt"
cfg.DATASET.TRAIN_INTRINSIC_PATH = None
cfg.DATASET.TRAIN_IMG_SIZE = (640, 480)

cfg.DATASET.VAL_DATA_SOURCE = "File"
cfg.DATASET.VAL_DATA_ROOT = "train_dataset_sample"
cfg.DATASET.VAL_NPZ_ROOT = "train_dataset_sample"
cfg.DATASET.VAL_LIST_PATH = "train_dataset_sample/drone_pairs_list.txt"
cfg.DATASET.VAL_INTRINSIC_PATH = None
cfg.DATASET.VAL_IMG_SIZE = (640, 480)

cfg.DATASET.FP16 = True
