import os
import time
import torch

# init
class Config:
    def __init__(self):
        # ------------------------------TRAIN------------------------
        self.SEED = 3035  # random seed, for reproduction
        self.DATASET = 'SHHA'  # dataset selection: SHHA, SHHB, SHHM
        self.NET = 'Res50'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet, Res101_SFCN
        self.PRE_GCC = False  # use the pretrained model on GCC dataset
        self.PRE_GCC_MODEL = 'path to model'  # path to model
        self.RESUME = False  # continue training
        self.RESUME_PATH = './exp/SHHB_ResNet/latest_state.pth'
        self.EXISTS_GPU = True
        self.GPU_ID = [0] if self.EXISTS_GPU else []  # single gpu: [0], [1] ...; multi gpus: [0,1]

        # learning rate settings
        self.LR = 1e-5  # learning rate
        self.LR_DECAY = 0.995 # decay rate
        self.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will begin to decay
        self.NUM_EPOCH_LR_DECAY = 1  # decay frequency
        self.MAX_EPOCH = 300

        # multi-task learning weights, no use for single model
        self.LAMBDA_1 = 1e-4  # SANet:0.001 CMTL 0.0001

        # print
        self.PRINT_FREQ = 10

        now = time.strftime("%m-%d_%H-%M", time.localtime())

        self.EXP_NAME = now \
                        + '_' + self.DATASET \
                        + '_' + self.NET \
                        + '_' + str(self.LR)

        self.EXP_PATH = '/graphics/scratch2/students/langstei/train_logs/exp'  # the path of logs, checkpoints, and current codes

        # ------------------------------VAL------------------------
        self.VAL_DENSE_START = 150
        self.VAL_FREQ = 10  # Before self.VAL_DENSE_START epoches, the freq is set as self.VAL_FREQ

        # ------------------------------VIS------------------------
        self.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with different sizes

# Create an instance of the Config class
cfg = Config()