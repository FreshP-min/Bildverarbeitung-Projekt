class ConfigSHHA:
    def __init__(self):
        self.STD_SIZE = (768, 1024)
        self.TRAIN_SIZE = (576, 768)  # 2D tuple or 1D scalar
        self.DATA_PATH = '/graphics/scratch2/students/langstei/ProcessedData/shanghaitech_part_A/'
        self.MEAN_STD = (
            [0.410824894905, 0.370634973049, 0.359682112932],
            [0.278580576181, 0.26925137639, 0.27156367898]
        )
        self.LABEL_FACTOR = 1
        self.LOG_PARA = 100.
        self.RESUME_MODEL = ''  # model path
        self.TRAIN_BATCH_SIZE = 8  # imgs
        self.VAL_BATCH_SIZE = 1  # must be 1

# Create an instance of the ConfigSHHA class
cfg_data = ConfigSHHA()
