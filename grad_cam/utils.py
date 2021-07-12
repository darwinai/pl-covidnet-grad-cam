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
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (224, 224), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    img = resize(img, (224, 224))
    cam_heatmap = cam_heatmap / np.max(cam_heatmap)
    cam_heatmap = np.float32(cam_heatmap) + np.float32(img)
    cam_heatmap = 255 * cam_heatmap / np.max(cam_heatmap)
    cam_heatmap = np.uint8(cam_heatmap)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(131)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')

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