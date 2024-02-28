from easydict import EasyDict as edict

# init
__C_SHHM = edict()

cfg_data = __C_SHHM

__C_SHHM.STD_SIZE = (768, 1024)
__C_SHHM.TRAIN_SIZE = (576, 768) # 2D tuple or 1D scalar
__C_SHHM.DATA_PATH = '/graphics/scratch2/students/langstei/ProcessedData/shanghaitech_merged/'

__C_SHHM.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])

__C_SHHM.LABEL_FACTOR = 1
__C_SHHM.LOG_PARA = 100.

__C_SHHM.RESUME_MODEL = ''#model path
__C_SHHM.TRAIN_BATCH_SIZE = 4 #imgs

__C_SHHM.VAL_BATCH_SIZE = 1 # must be 1


