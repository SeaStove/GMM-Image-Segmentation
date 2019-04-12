import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
from sklearn import preprocessing
import pandas as pd
from PIL import Image
import os
import os.path

# Can be many different formats.
image_path = ('./data-set/train/birds/bird1.png')


def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""

    png = Image.open(image_path)

    image = Image.new("RGB", png.size, (255, 255, 255))
    image.paste(png, mask=png.split()[3])

    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width * height, 3))
    # pixel_values.shape = (width * height, 3)

    return pixel_values


def get_pixel(image_array):
    print(image_array[0])
    return 0


def get_GMM(rgb_array):
    gmm = GaussianMixture(n_components=2)
    # print(np.cov(rgb_array).shape)
    gmm.fit(rgb_array, rgb_array.shape)
    # print(gmm.means_)
    # print(gmm.covariances_)
    # print(gmm.weights_)
    gmm.predict(rgb_array)

    return gmm


def get_all_pixels(directory, image_type):
    num_images = len([name for name in os.listdir(directory)
                      if os.path.isfile(os.path.join(directory, name))])
    images_array = get_image(directory + image_type + "1.png")
    for i in range(num_images - 2):
        new_array = get_image(directory + image_type + str(i+2) + ".png")
        image_array = np.append(images_array, new_array, axis=0)

    return image_array


def predict_GMM(image_path, gmm):
    image_array = get_image(image_path)
    print(image_array.shape)
    prediction = gmm.predict(image_array[0].reshape(1, 3))
    return prediction


bird_array = get_all_pixels("./data-set/train/birds/", "bird")
print("Shape of bird: ", bird_array.shape)
sky_array = get_all_pixels("./data-set/train/sky/", "sky")
print("Shape of sky: ", sky_array.shape)
both_array = np.append(sky_array, bird_array, axis=0)
print("Shape of both: ", both_array.shape)
# print(image_array.shape)
GMM = get_GMM(both_array)
print(predict_GMM("./data-set/train/birds/bird25.png", GMM))
print(predict_GMM("./data-set/train/sky/sky1.png", GMM))
