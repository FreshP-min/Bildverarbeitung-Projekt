import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from config import cfg
import torchvision.transforms as transforms
from models.CC import CrowdCounter
import ast

# trained model and dataset to make predictions on
model_name = '01-23_14-53_SHHB_Res50_1e-05'
dataset_name = 'SHHA'

# model path
model_path = 'Trained_models/' + model_name + '/best_model.pth'

# path to dataset, gt density maps and images
dataset_path = 'datasets/ProcessedData/shanghaitech_part_A/test'
den_path = dataset_path + '/den'
img_path = dataset_path + '/img'

# path to .txt file to save ground truth labels
save_file_gt = dataset_path + '/gt_number.txt'

# path to .txt file to save predicted labels
save_file_pred = dataset_path + '/pred_number_' + model_name + '.txt'

# path to save plot
plot_path = '../Auswertungen/' + model_name + '_on_' + dataset_name

def plot_sample(true_labels, pred_labels, max_val=10000):

    # use this function to generate a plot with true labels on x axis and predicted labels on y axis
    # input: list of true labels, list of predicted labels
    # returns plot

    ys = []
    yhats = []
    for x, y in zip(true_labels, pred_labels):
        if (x <= max_val):
            ys.append(x)
            yhats.append(y)

    plt.figure(figsize=(6, 6))
    plt.scatter(ys, yhats, alpha=0.4)
    plt.plot([min(ys), max(ys)], [min(ys), max(ys)], 'k--', lw=4)
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title(model_name + ' on ' + dataset_name)
    return plt

def get_gt_number():

    # generate a list of true labels (the number of people in each picture) from a dataset
    # labels are saved to be able to reuse them for different plots

    # generate list of all density map files
    file_list_den = [os.path.join(den_path, filename) for root, dirs, filenames in os.walk(den_path) for filename in filenames]

    true_labels = []

    # compute number of people for every density map
    for file in file_list_den:
        den = pd.read_csv(file, sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        gt = np.sum(den)
        true_labels.append(gt)

    # print labels to be able to save them manually if saving list did not work
    print(true_labels)

    # save ground truth labels
    with open(save_file_gt, 'w') as save_file:
        save_file.write(str(true_labels))


def get_pred_number():

    # generate a list of predicted labels (the number of people in each picture) for a dataset
    # this could take some time at least on CPU since it computes one density map prediction for each picture
    # labels are saved to be able to reuse them for different plots

    # generate list of all image files
    file_list_img = [os.path.join(img_path, filename) for root, dirs, filenames in os.walk(img_path) for filename in filenames]

    # load model
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    if (cfg.EXISTS_GPU):
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()

    pred_labels = []

    # inference on all images
    num = 1
    for imgname in file_list_img:
        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        img = img_transform(img)

        with torch.no_grad():
            if cfg.EXISTS_GPU:
                img = img[None, :, :, :].cuda()
            else:
                img = img[None, :, :, :]
            pred_map = net.test_forward(img)
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        # get number of people
        pred = np.sum(pred_map) / 100.0
        pred = round(pred, 5)
        print(f"prediction {num} : {pred}")
        pred_labels.append(pred)
        num = num + 1

    # print labels to be able to save them manually if saving list did not work
    print(pred_labels)

    # save predicted labels
    with open(save_file_pred, 'w') as save_file:
        save_file.write(str(pred_labels))


if __name__ == '__main__':

    # generate lists of labels if not generated yet
    #get_gt_number()
    #get_pred_number()

#"""""
    # take the previously generated lists of gt and predicted labels
    gt_number = []
    pred_number = []

    with open(save_file_gt, 'r') as gt_list:
        gt_number = gt_list.readline()

    with open(save_file_pred, 'r') as pred_list:
        pred_number = pred_list.readline()

    gt_number = ast.literal_eval(gt_number)
    pred_number = ast.literal_eval(pred_number)
    print(gt_number)
    print(pred_number)

    # make the plot, save and show it
    plot = plot_sample(gt_number, pred_number)
    plot.savefig(plot_path)
    plot.show()
    
#"""""