# DS_NAME = 'gland'
# IM_WIDTH = 480
# IM_HEIGHT = 480
# IM_PAD_WIDTH = 784
# IM_PAD_HEIGHT = 528
# IM_CHANNEL = 3
# N_UNLABELED = 85
# N_SELECT = 5
# BATCH_SIZE = 5
# training_data_path = "DATA/Data/glanddata.npy"
# val_data_path = None
# test_data_path = "DATA/Data/glanddata_testb.npy"

DS_NAME = 'QB'
IM_WIDTH = 256
IM_HEIGHT = 256
IM_PAD_WIDTH = 256
IM_PAD_HEIGHT = 256
IM_CHANNEL = 6
N_UNLABELED = 1350
N_SELECT = 67
BATCH_SIZE = 128
training_data_path = "DATA/Data/QB_train.npy"
val_data_path = "DATA/Data/QB_val.npy"
test_data_path = "DATA/Data/QB_test_benign.npy"

MAX_RUN_COUNT = 1
OUTPUT_DIR = 'output'
INIT_N_EPOCH = 20
MAX_N_EPOCH = 1300
VAL_STEP = 10
