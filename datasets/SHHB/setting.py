class ConfigSHHB:
    def __init__(self):
        self.STD_SIZE = (768, 1024)
        self.TRAIN_SIZE = (576, 768)
        self.DATA_PATH = '/graphics/scratch2/students/langstei/ProcessedData/shanghaitech_part_B/'
        self.MEAN_STD = (
            [0.452016860247, 0.447249650955, 0.431981861591],
            [0.23242045939, 0.224925786257, 0.221840232611]
        )
        self.LABEL_FACTOR = 1
        self.LOG_PARA = 100.
        self.RESUME_MODEL = '../../models/SCC_Model/Res50.py'  # model path
        self.TRAIN_BATCH_SIZE = 4  # imgs
        self.VAL_BATCH_SIZE = 1

# Create an instance of the ConfigSHHB class
cfg_data = ConfigSHHB()
