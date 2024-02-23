import matplotlib.pyplot as plt
import Prepocessing
pre = Prepocessing.Preprocessing()

img = plt.imread("1.jpg")
'''
for i in range(1, 301):
    pre.split_and_process(i, 2)
'''
'''
for i in range(5):
    a = pre.create_random_blur_kernel()

    plt.imshow(a)
    plt.show()
'''

b = pre.convert_to_grayscale(img)
plt.imshow(b, cmap="gray")
plt.show()