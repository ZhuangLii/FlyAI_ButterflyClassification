from yacs.config import CfgNode as CN
_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'resnest50'
_C.MODEL.METRIC_LOSS_TYPE = 'ce_center'
_C.MODEL.IF_LABELSMOOTH = 'on'
_C.MODEL.POOLING_METHOD = 'GeM'
_C.MODEL.ID_LOSS_TYPE = ''
_C.MODEL.FEAT_SIZE = 2048
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.BATCHDROP = False
_C.MODEL.LAST_STRIDE = 2
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.DAULPATH = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [224, 224]
# Size of the image during test
_C.INPUT.SIZE_TEST = [224, 224]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.MIXUP = False
# Value of padding size
_C.INPUT.PADDING = 10
_C.INPUT.GRID_PRO = 0.0
_C.INPUT.GRAY_RPO = 0.05

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = ('./data/input/ButterflyClassification/')
_C.DATASETS.HARD_AUG = 'auto'
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 0.001
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.CENTER_LR = 0.2
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0003

_C.SOLVER.T_MAX = 5
_C.SOLVER.ETA_MIN = 0.001
_C.SOLVER.SWA = False
_C.SOLVER.SWA_START = [50, 60, 70, 80]
_C.SOLVER.SWA_ITER = 10
_C.SOLVER.SWA_MAX = 0.0045
_C.SOLVER.SWA_MIN = 0.001
_C.SOLVER.GRADUAL_UNLOCK = False
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (25, 40)

_C.SOLVER.TYPE = 'warmup'
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"


_C.SOLVER.COSINE_MARGIN = 0.1
_C.SOLVER.COSINE_SCALE = 40


# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
_C.SOLVER.FP16 = False
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.SEED = 1234
_C.SOLVER.GRADCENTER = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./log/"
_C.TBOUTPUT_DIR = "./log/tensorboard"
_C.IF_VAL = False