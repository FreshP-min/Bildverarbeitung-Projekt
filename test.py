from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

if cfg.EXISTS_GPU:
    torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
with_gt = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_name = '../SHHB_results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if with_gt:
    if not os.path.exists(exp_name + '/pred'):
        os.mkdir(exp_name + '/pred')

    if not os.path.exists(exp_name + '/gt'):
        os.mkdir(exp_name + '/gt')

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])

img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])

dataRoot = 'datasets/ProcessedData/shanghaitech_part_B/test'

model_path = 'exp/01-10_15-29_SHHB_MCNN_1e-05/all_ep_1_mae_71.0_mse_96.0.pth'


def main():
    if with_gt:
        file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/img/')]
    else:
        file_list = [filename for root, dirs, filename in os.walk(dataRoot)]
    test(file_list[0], model_path)


def test(file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    if (cfg.EXISTS_GPU):
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    net.eval()

    f1 = plt.figure(1)

    gts = []
    preds = []

    for filename in file_list:
        print(filename)
        if(with_gt):
            imgname = dataRoot + '/img/' + filename
            filename_no_ext = filename.split('.')[0]

            denname = dataRoot + '/den/' + filename_no_ext + '.csv'

            den = pd.read_csv(denname, sep=',', header=None).values
            den = den.astype(np.float32, copy=False)
            gt = np.sum(den)

        else:
            imgname = dataRoot + filename
            filename_no_ext = filename.split('.')[0]

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)

        with torch.no_grad():
            if cfg.EXISTS_GPU:
                img = img[None, :, :, :].cuda()
            else:
                img = img[None, :, :, :]
            pred_map = net.test_forward(img)

        sio.savemat(exp_name + '/pred/' + filename_no_ext + '.mat',
                    {'ProcessedData': pred_map.squeeze().cpu().numpy() / 100.})
        if with_gt:
            sio.savemat(exp_name + '/gt/' + filename_no_ext + '.mat', {'ProcessedData': den})

        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        pred = np.sum(pred_map) / 100.0
        print(f"prediction: {pred}")
        pred_map = pred_map / np.max(pred_map + 1e-20)

        if with_gt:
            den = den / np.max(den + 1e-20)

            den_frame = plt.gca()
            plt.imshow(den, 'jet')
            den_frame.axes.get_yaxis().set_visible(False)
            den_frame.axes.get_xaxis().set_visible(False)
            den_frame.spines['top'].set_visible(False)
            den_frame.spines['bottom'].set_visible(False)
            den_frame.spines['left'].set_visible(False)
            den_frame.spines['right'].set_visible(False)
            plt.savefig(exp_name + '/' + filename_no_ext + '_gt_' + str(int(gt)) + '.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'ProcessedData':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig(exp_name + '/' + filename_no_ext + '_pred_' + str(float(pred)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'ProcessedData':pred_map})


        if with_gt:
            diff = den - pred_map

            diff_frame = plt.gca()
            plt.imshow(diff, 'jet')
            plt.colorbar()
            diff_frame.axes.get_yaxis().set_visible(False)
            diff_frame.axes.get_xaxis().set_visible(False)
            diff_frame.spines['top'].set_visible(False)
            diff_frame.spines['bottom'].set_visible(False)
            diff_frame.spines['left'].set_visible(False)
            diff_frame.spines['right'].set_visible(False)
            plt.savefig(exp_name + '/' + filename_no_ext + '_diff.png', \
                        bbox_inches='tight', pad_inches=0, dpi=150)

            plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'ProcessedData':diff})

def load_model(path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    if (cfg.EXISTS_GPU):
        net.load_state_dict(torch.load(path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    net.eval()

    print('model loaded')
    return net

def apply_counting(img, net):

    if img.mode == 'L':
        img = img.convert('RGB')

    old_size = (img.size[0], img.size[1])
    if (old_size[0] and old_size[1]) > 1000:
        new_size = (896, 896)
    elif (old_size[0] and old_size[1]) > 500:
        new_size = (448, 448)
    else:
        new_size = old_size
    img = img.resize((new_size[0], new_size[1]), Image.BILINEAR)

    img = img_transform(img)

    with torch.no_grad():
        if cfg.EXISTS_GPU:
            img = img[None, :, :, :].cuda()
        else:
            img = img[None, :, :, :]
        pred_map = net.test_forward(img)

    pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

    pred = np.sum(pred_map) / 100.0
    pred = round(pred, 2)
    print(f"prediction: {pred}")
    pred_map = pred_map / np.max(pred_map + 1e-20)
    plt.imshow(pred_map)
    #   plt.show()
    pred_map_pil = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(pred_map) * 255))
    pred_map_pil = pred_map_pil.resize((old_size[0], old_size[1]), Image.BILINEAR)

    return pred_map_pil, pred


if __name__ == '__main__':
    img = Image.open("datasets/ProcessedData/shanghaitech_part_B/test_after_training/img/14.jpg")
    apply_counting(img)
    #main()
