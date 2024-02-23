import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpl_image
import scipy.ndimage as ndimage
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torch
class Preprocessing():
    def __init__(self):
        pass

    '''
    to do:
    - create fast convolution function, maybe also in frequency space
    '''

    # applying gaussian filter on image
    def gaussian_2d(self, x, y, sigma):
        return (1 / 2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


    def add_noise(self, img: np.ndarray, mean: float, sigma: float) -> np.ndarray:
        height, width = img.shape[0], img.shape[1]
        noise = np.random.normal(mean, sigma, (height, width))
        noise /= np.sum(noise)
        res = np.zeros(img.shape, dtype=np.int64)
        res[:, :, 0] = np.clip(img[:, :, 0] + noise, 0, 255)
        res[:, :, 1] = np.clip(img[:, :, 1] + noise, 0, 255)
        res[:, :, 2] = np.clip(img[:, :, 2] + noise, 0, 255)
        return res


    def padding(self, img: np.ndarray, kernel_size: int) -> np.ndarray:
        pad_size = kernel_size // 2
        height, width = img.shape[0], img.shape[1]
        padded_img = np.zeros((height + 2 * pad_size, width + 2 * pad_size), dtype=np.int64)
        for i in range(height):
            for j in range(width):
                padded_img[i + pad_size][j + pad_size] = img[i][j]

        for j in [pad_size, width + pad_size - 1]:
            if j == pad_size:
                for i in range(pad_size, height + pad_size):
                    for k in range(1, pad_size + 1):
                        padded_img[i][j - k] = padded_img[i][j + k - 1]
            if j == width + pad_size - 1:
                for i in range(pad_size, height + pad_size):
                    for k in range(1, pad_size + 1):
                        padded_img[i][j + k] = padded_img[i][j - k + 1]

        for i in [pad_size, height + pad_size - 1]:
            if i == pad_size:
                for j in range(pad_size, width + pad_size):
                    for k in range(1, pad_size + 1):
                        padded_img[i - k][j] = padded_img[i + k - 1][j]
            if i == height + pad_size - 1:
                for j in range(pad_size, width + pad_size):
                    for k in range(1, pad_size + 1):
                        padded_img[i + k][j] = padded_img[i - k + 1][j]

        # edges
        for i in range(pad_size):
            for j in range(pad_size):
                padded_img[i][j] = padded_img[i][2 * pad_size - j]
                padded_img[i][width + 2 * pad_size - 1 - j] = padded_img[i][width + j]
                padded_img[height + pad_size + i][j] = padded_img[height + pad_size - 1 - i][j]
                padded_img[height + pad_size + i][width + 2 * pad_size - 1 - j] = padded_img[height + pad_size - 1 - i][width+1+j]

        return padded_img



    def convolve(self, img: np.ndarray, color: bool, conv_f, sigma: float, mean: float, kernel_size: int) -> np.ndarray:
        convolved_img = np.zeros(img.shape)
        height, width = img.shape[0], img.shape[1]
        pad_size = kernel_size // 2
        if color:
            padded_img = np.zeros(img.shape)
            for i in range(3):
                padded_img[:, :, i] = self.padding(img[:, :, i], kernel_size)
            for x in range(width):
                for y in range(height):
                    val = [0, 0, 0]
                    for i in range(3):
                        for k_x in range(-pad_size, pad_size+1):
                            for k_y in range(-pad_size, pad_size+1):
                                val[i] += conv_f(k_x, k_y, sigma) * padded_img[y+k_y, x+k_x, i]
                        convolved_img[y][x][i] = (sigma/kernel_size*kernel_size) * val[i]
        else:
            padded_img = self.padding(img, kernel_size)
            for x in range(width):
                for y in range(height):
                    val = 0
                    for k_x in range(-pad_size, pad_size+1):
                        for k_y in range(-pad_size, pad_size+1):
                            val += conv_f(k_x, k_y, sigma) * padded_img[y+k_y, x+k_x]
                        convolved_img[y][x] = (sigma/kernel_size*kernel_size) * val

        return convolved_img




    # image partition into several parts
    def split_den_map(self, number: int, n: int):
        with open(f"{number}.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            den_map = [row for row in csv_reader]
            img = plt.imread(f"{number}.jpg")
            height, width = img.shape[0], img.shape[1]
            split_maps = []
            split_imgs = []
            for i in range(0, n):
                for j in range(0, n):
                    split_maps.append([row[i*(width//n):(i+1)*(width//n)] for row in den_map[j*(height//n):(j+1)*(height//n)]])
                    split_imgs.append([row[i * (width // n):(i + 1) * (width // n)] for row in img[j * (height // n):(j + 1) * (height // n), :]])
                    mpl_image.imsave(f"{number}_{i}_{j}.jpg", split_imgs[i][j])
                    with open(f'{number}_{i}_{j}.csv', 'x') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerows(split_maps[i][j])



    def blur_image(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape[0], img.shape[1]
        ran_blur = self.create_random_blur_kernel()
        ran_blur /= np.sum(ran_blur)

        if len(img.shape) == 3:
            res = np.zeros((height, width, 3))
            for k in range(3):
                res[:, :, k] = ndimage.convolve(img[:, :, k], ran_blur)
        else:
            res = ndimage.convolve(img, ran_blur)

        return img


    def split_and_process(self, number: int, split_n: int):
        with open(f"shanghaitech_part_A/train/den/{number}.csv") as open_f:
            csv_reader = csv.reader(open_f, delimiter=',')
            den_map = [row for row in csv_reader]
            img = plt.imread(f"shanghaitech_part_A/train/img/{number}.jpg")
            height, width = img.shape[0], img.shape[1]
            #ran_blur = plt.imread("blurkernel.png")
            #ran_blur /= np.sum(ran_blur)
            ran_blur = self.create_random_blur_kernel()
            ran_blur /= np.sum(ran_blur)
            if len(img.shape) == 3:
                for i in range(0, split_n):
                    for j in range(0, split_n):
                        split_map = [row[i*(width//split_n):(i+1)*(width//split_n)] for row in den_map[j*(height//split_n):(j+1)*(height//split_n)]]
                        split_img = np.array([row[i * (width // split_n):(i + 1) * (width // split_n)] for row in img[j * (height // split_n):(j + 1) * (height // split_n), :]])

                        for k in range(0, 3):
                            split_img[:, :, k] = ndimage.convolve(split_img[:, :, k], ran_blur)
                        mpl_image.imsave(f"processed/img/{number}_{i}_{j}.jpg", split_img)
                        with open(f'processed/den/{number}_{i}_{j}.csv', 'x') as f:
                            writer = csv.writer(f, delimiter=',')
                            writer.writerows(split_map)
            else:
                for i in range(0, split_n):
                    for j in range(0, split_n):
                        split_map = [row[i * (width // split_n):(i + 1) * (width // split_n)] for row in
                                     den_map[j * (height // split_n):(j + 1) * (height // split_n)]]
                        split_img = np.array([row[i * (width // split_n):(i + 1) * (width // split_n)] for row in
                                              img[j * (height // split_n):(j + 1) * (height // split_n)]])
                        split_img = ndimage.convolve(split_img, ran_blur)
                        mpl_image.imsave(f"processed/img/{number}_{i}_{j}.jpg", split_img, cmap="gray")
                        with open(f'processed/den/{number}_{i}_{j}.csv', 'x') as f:
                            writer = csv.writer(f, delimiter=',')
                            writer.writerows(split_map)



    def create_random_blur_kernel(self) -> np.ndarray:
        blur_pixels = np.zeros((15, 15))
        rand_inds = [(7, 7)]
        for i in range(15):
            x = rand_inds[i][0]
            y = rand_inds[i][1]
            rand_x, rand_y = np.random.randint(x - 1 if x-1 >= 0 else 0, x + 2 if x+2 <= 14 else 14), np.random.randint(y - 1 if y-1 >= 0 else 0, y + 2 if y+2 <= 14 else 14)
            while (rand_x, rand_y) in rand_inds:
                rand_x, rand_y = np.random.randint(x-1, x+2), np.random.randint(y-1, y+2)
            rand_inds.append((rand_x, rand_y))
            blur_pixels[x][y] = 255
        blur_kernel = self.convolve(blur_pixels, False, self.gaussian_2d, 1, 0, 5)
        return blur_kernel

    def random_crop(self, img: np.ndarray):
        cropper = v2.RandomCrop(size=(64, 64))
        #crops = [cropper(img) for _ in range(4)]
        plt.imshow(transforms.RandomCrop(size=64))
        plt.show()

    def vertical_flip(self, img: np.ndarray) -> np.ndarray:
        return ndimage.rotate(img, 180, reshape=False)

    # very slow, might be used with caution
    def convert_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        res = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
        res[:, :] = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        return res