b = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
c = np.copy(b)
#plt.imshow(c, cmap="gray")
#plt.show()
sigma = 1
for i in range(1, height - 1):
    for j in range(1, width - 1):
        val = 0
        for k1 in range(-1, 2):
            for k2 in range(-1, 2):
                val += gaussian_2d(k1, k2, sigma) * b[k1 + i][k2 + j]
        c[i][j] = (sigma / 9) * val

#plt.imshow(c, cmap="gray")
#plt.show()

# possible method of creating blur on image





blur_kernel = plt.imread("blurkernel.png")
blur_kernel_dims = blur_kernel.shape
blur_kernel /= np.sum(blur_kernel)

e = ndimage.convolve(b, blur_kernel)

mean = 0
sigma = 20


r = add_noise(img, mean, sigma)
plt.imshow(r)
plt.show()

#plt.imshow(impulse_img, cmap='gray')
#plt.show()


impulse_img = np.zeros((16, 16), dtype=np.int64)
impulse_img[8][8] = 255;


sigma = 1
kernel_size = 5
gaussian_kernel = np.zeros((16, 16))

impulse_img = padding(impulse_img, 5)

#plt.imshow(impulse_img, cmap="gray")
#plt.show()
pad_size = kernel_size//2
for i in range(pad_size, 16+pad_size-1):
    for j in range(pad_size, 16+pad_size-1):
        val = 0
        for k1 in range(-2, 3):
            for k2 in range(-2, 3):
                val += gaussian_2d(k1, k2, sigma) * impulse_img[k1 + i][k2 + j]
        gaussian_kernel[i-pad_size][j-pad_size] = val

gaussian_kernel /= np.sum(gaussian_kernel)
#plt.imshow(gaussian_kernel, cmap="gray")
#plt.show()

blurred_image1 = np.zeros_like(img)

blurred_image2 = np.zeros_like(img)
import scipy.ndimage

for i in range(3):
    blurred_image1[:, :, i] = ndimage.convolve(img[:, :, i], gaussian_kernel)
    blurred_image2[:, :, i] = ndimage.convolve(img[:, :, i], blur_kernel)



img = plt.imread("shopping_mall.jpg")
    plt.imshow(img, cmap="gray")
    '''
    a = np.array(img)
    b = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i][j] = 0.2126 * a[i][j][0] + 0.7152 * a[i][j][1] + 0.0722 * a[i][j][2]

    #plt.imshow(a, cmap="gray")
    #plt.imshow(b, cmap="gray")
    #plt.show()
    #print(img.shape)
    '''
    height, width = img.shape[0], img.shape[1]
