import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpl_image
import scipy.ndimage as ndimage



def gaussian_2d( x, y, sigma):
    return (1 / 2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def add_noise(img: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    height, width = img.shape[0], img.shape[1]
    noise = np.random.normal(mean, sigma, (height, width))
    res = np.zeros(img.shape, dtype=np.int64)
    res[:, :, 0] = np.clip(img[:, :, 0] + noise, 0, 255)
    res[:, :, 1] = np.clip(img[:, :, 1] + noise, 0, 255)
    res[:, :, 2] = np.clip(img[:, :, 2] + noise, 0, 255)
    return res


def padding(img: np.ndarray, kernel_size: int) -> np.ndarray:
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
            padded_img[height + pad_size + i][width + 2 * pad_size - 1 - j] = padded_img[height + pad_size - 1 - i][
                width + 1 + j]

    return padded_img


def convolve(img: np.ndarray, color: bool, conv_f, sigma: float, mean: float, kernel_size: int) -> np.ndarray:
    convolved_img = np.zeros(img.shape)
    height, width = img.shape[0], img.shape[1]
    pad_size = kernel_size // 2
    if color:
        padded_img = np.zeros(img.shape)
        for i in range(3):
            padded_img[:, :, i] = padding(img[:, :, i], kernel_size)
        for x in range(width):
            for y in range(height):
                val = [0, 0, 0]
                for i in range(3):
                    for k_x in range(-pad_size, pad_size + 1):
                        for k_y in range(-pad_size, pad_size + 1):
                            val[i] += conv_f(k_x, k_y, sigma) * padded_img[y + k_y, x + k_x, i]
                    convolved_img[y][x][i] = (sigma / kernel_size * kernel_size) * val[i]
    else:
        padded_img = padding(img, kernel_size)
        for x in range(width):
            for y in range(height):
                val = 0
                for k_x in range(-pad_size, pad_size + 1):
                    for k_y in range(-pad_size, pad_size + 1):
                        val += conv_f(k_x, k_y, sigma) * padded_img[y + k_y, x + k_x]
                    convolved_img[y][x] = (sigma / kernel_size * kernel_size) * val

    return convolved_img


# image partition into several parts
def split_den_map(number: int, n: int):
    with open(f"{number}.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        den_map = [row for row in csv_reader]
        img = plt.imread(f"{number}.jpg")
        height, width = img.shape[0], img.shape[1]
        for i in range(0, n):
            for j in range(0, n):
                mpl_image.imsave(f"{number}_{i}_{j}.jpg", np.array([row[i * (width // n):(i + 1) * (width // n)] for row in
                                   img[j * (height // n):(j + 1) * (height // n), :]]))
                with open(f'{number}_{i}_{j}.csv', 'x') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows(np.array([row[i * (width // n):(i + 1) * (width // n)] for row in
                                   den_map[j * (height // n):(j + 1) * (height // n)]]))


def split_and_process(number, split_n: int):
    with open(f"shanghaitech_part_A/train/den/{number}.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        den_map = [row for row in csv_reader]
        img = plt.imread(f"shanghaitech_part_A/train/img/{number}.jpg")
        height, width = img.shape[0], img.shape[1]
        print(height)
        print(width)
        split_maps = []
        split_imgs = []
        for i in range(0, split_n):
            for j in range(0, split_n):
                split_maps.append([row[i * (width // split_n):(i + 1) * (width // split_n)] for row in
                                   den_map[j * (height // split_n):(j + 1) * (height // split_n)]])
                split_imgs.append([row[i * (width // split_n):(i + 1) * (width // split_n)] for row in
                                   img[j * (height // split_n):(j + 1) * (height // split_n), :]])
                # split_imgs[i][j] = ndimage.convolve(split_imgs[i][j], self.create_random_blur_kernel())
                mpl_image.imsave(f"{number}_{i}_{j}.jpg", split_imgs[i][j])
                with open(f'shanghaitech_part_A/train/den/{number}_{i}_{j}.csv', 'x') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows(split_maps[i][j])


def create_random_blur_kernel() -> np.ndarray:
    blur_pixels = np.zeros((15, 15))
    rand_inds = [(7, 7)]
    for i in range(4):
        x = rand_inds[i][0]
        y = rand_inds[i][1]
        rand_x, rand_y = np.random.randint(x - 1, x + 2), np.random.randint(y - 1, y + 2)
        while rand_x == x and rand_y == y:
            rand_x, rand_y = np.random.randint(x - 1, x + 2), np.random.randint(y - 1, y + 2)
        rand_inds.append((rand_x, rand_y))
        blur_pixels[x][y] = 255
    blur_kernel = convolve(blur_pixels, False, gaussian_2d, 1, 0, 5)
    return blur_kernel