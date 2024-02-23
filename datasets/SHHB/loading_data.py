import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .SHHB import SHHB
from .setting import cfg_data
import torch


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
        own_transforms.RandomHorizontallyFlip()
    ])
    val_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop(cfg_data.TRAIN_SIZE)
    ])
    val_main_transform = None
    img_transform = standard_transforms.Compose([
        standard_transforms.RandomChoice([
            own_transforms.Blur(1),
            own_transforms.Noise(1),
            own_transforms.VerticalFlip(1)
            # own_transforms.Grayscale(1)
        ], p=0.25),
        standard_transforms.Normalize(*mean_std)
    ])
    val_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = SHHB(cfg_data.DATA_PATH + '/train', 'train', main_transform=train_main_transform,
                     img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True,
                              drop_last=True)

    print('train data loaded')

    val_set = SHHB(cfg_data.DATA_PATH + '/test', 'test', main_transform=val_main_transform,
                   img_transform=val_img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    print('test data loaded')

    return train_loader, val_loader, restore_transform
