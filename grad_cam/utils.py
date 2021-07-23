import skimage
import skimage.io
import skimage.transform
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg

import tensorflow as tf

from skimage import io
from skimage.transform import resize

import cv2


def visualize(image, conv_output, conv_grad):
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]
    print("grads_val shape:", grads_val.shape)

    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
    cam = np.zeros(output.shape[0:2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam2 = cam
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (480, 480), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    cam_heatmap = cam_heatmap / np.max(cam_heatmap)
    cam_heatmap = np.float32(cam_heatmap) + np.float32(img)
    cam_heatmap = 255 * cam_heatmap / np.max(cam_heatmap)
    cam_heatmap = np.uint8(cam_heatmap)

    cam2 -= np.min(cam2)
    cam2 = cam2 / np.max(cam2)
    cam2 = resize(cam2, (480, 480), preserve_range=True)
    _, cam_heatmap_2 = cv2.threshold(np.uint8(255 * cam2), 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cam_heatmap_2 = cam_heatmap_2 / np.max(cam_heatmap_2)
    cam_heatmap_2 = 255 - cam_heatmap_2
    cam_heatmap_2 = cv2.cvtColor(cam_heatmap_2, cv2.COLOR_GRAY2RGB)
    purple = [104, 54, 169]
    for i in range(3):
        cam_heatmap_2[:, :, i] = cam_heatmap_2[:, :, i] / 255 * purple[i]
    # edges = cv2.Canny(np.uint8(255 * cam_heatmap_2), 0, 255)
    # # cv2 channels formated as bgr
    # from copy import deepcopy
    # outlined_image = deepcopy(np.uint8(255 * img))
    # outlined_image[:, :, 2] += edges
    # outlined_image[outlined_image > 255] = 255

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(131)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(151)
    imgplot = plt.imshow(cam_heatmap_2, vmin=0, vmax=255)
    ax.set_title('Grad-CAM with Threshold')

    # gb_viz = np.dstack((
    #     gb_viz[:, :, 0],
    #     gb_viz[:, :, 1],
    #     gb_viz[:, :, 2],
    # ))
    # gb_viz -= np.min(gb_viz)
    # gb_viz /= gb_viz.max()

    # ax = fig.add_subplot(132)
    # imgplot = plt.imshow(gb_viz)
    # ax.set_title('guided backpropagation')

    # gd_gb = np.dstack((
    #     gb_viz[:, :, 0] * cam,
    #     gb_viz[:, :, 1] * cam,
    #     gb_viz[:, :, 2] * cam,
    # ))
    # ax = fig.add_subplot(133)
    # imgplot = plt.imshow(gd_gb)
    # ax.set_title('guided Grad-CAM')

    plt.show()