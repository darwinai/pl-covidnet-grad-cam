import tensorflow as tf
import os

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.applications.vgg16 import (VGG16, preprocess_input,
                                      decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Model
from numpy.lib.function_base import place
from tensorflow.python.framework import ops
import keras.backend as K

import numpy as np
import keras
import sys
import cv2


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


# def load_image(path):
#     img = image.load_img(path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [
        grad if grad is not None else tf.zeros_like(var)
        for var, grad in zip(var_list, grads)
    ]


def grad_cam(graph, image, category_index, layer_name):
    labels_tensor = graph.get_tensor_by_name('norm_dense_1_target:0')
    sample_weights = graph.get_tensor_by_name('norm_dense_1_sample_weights:0')
    pred_tensor = graph.get_tensor_by_name('norm_dense_1/Softmax:0')
    in_tensor = graph.get_tensor_by_name('input_1:0')
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred_tensor, labels=labels_tensor) * sample_weights)
    conv_output = graph.get_tensor_by_name('conv5_block3_out')
    grads = normalize(_compute_gradients(loss_op, [conv_output])[0])
    gradient_function = K.function([in_tensor], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    #shift values by min
    cam -= np.min(cam)
    # cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    _, cam = cv2.threshold(np.uint8(255 * heatmap), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cam = cam / np.max(cam)
    cam = 1 - cam
    edges = cv2.Canny(np.uint8(255 * cam), 0, 255)
    # cv2 channels formated as bgr
    from copy import deepcopy
    outlined_image = deepcopy(image)
    outlined_image[:, :, 2] += edges
    outlined_image[outlined_image > 255] = 255

    # masked_image = image*cam.reshape((224,224, 1))
    # cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    #cam = np.float32(cam) #+  np.float32(image)
    #cam = cam / np.max(cam)

    return np.uint8(outlined_image), heatmap


# model = VGG16(weights='imagenet')

# predictions = model.predict(preprocessed_input)
# print(predictions)
# top_1 = decode_predictions(predictions)[0][0]
# print(decode_predictions(predictions))
# print(f'Predicted class: {predicted_class}')
# print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

# predicted_class = np.argmax(predictions)
import os
import json
from data import process_image_file
import utils


def readInput():
    res = []
    res.append(process_image_file(sys.argv[1], 0.08, 480))

    with open(sys.argv[2], 'r') as f:
        input = json.load(f)
        res.append(
            np.array([
                float(x) for x in
                [input['Normal'], input['Pneumonia'], input['COVID-19']]
            ]))
    return res[0], res[1]


preprocessed_input, prediction_matrix = readInput()

print(np.shape(prediction_matrix))
print(prediction_matrix)

mapping = {'Normal': 0, 'Pneumonia': 1, 'COVID-19': 2}

tf.reset_default_graph()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    args = {
        'weightspath': os.path.abspath('models/COVIDNet-CXR4-B'),
        'ckptname': 'model-1545',
        'modelused': 'modelB'
    }
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(
        os.path.join(args['weightspath'], 'model.meta'))
    saver.restore(sess, os.path.join(args['weightspath'], args['ckptname']))

    graph = tf.get_default_graph()
    placeholders = [
        op for op in graph.get_operations() if op.type == "Placeholder"
    ]
    print(placeholders)
    image_tensor = graph.get_tensor_by_name('input_1:0')
    pred_tensor = graph.get_tensor_by_name('norm_dense_1/Softmax:0')

    x = preprocessed_input
    x = x.astype('float32') / 255.0
    pred = sess.run(pred_tensor,
                    feed_dict={image_tensor: np.expand_dims(x, axis=0)})
    print(pred)
    labels_tensor = graph.get_tensor_by_name('norm_dense_1_target:0')
    sample_weights = graph.get_tensor_by_name('norm_dense_1_sample_weights:0')
    pred_logit_tensor = graph.get_tensor_by_name('norm_dense_1/MatMul:0')
    pred_softmax_tensor = graph.get_tensor_by_name('norm_dense_1/Softmax:0')
    # image_tensor = graph.get_tensor_by_name('input_1:0')
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency
    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred_logit_tensor, labels=labels_tensor) * sample_weights)
    conv_output = graph.get_tensor_by_name('conv5_block3_out/add:0')
    target_conv_layer_grad = tf.gradients(loss_op, conv_output)[0]
    one_hot = np.zeros_like(prediction_matrix)
    one_hot[np.argmax(prediction_matrix)] = 1

    target_conv_layer_value, target_conv_layer_grad_value = sess.run(
        [conv_output, target_conv_layer_grad],
        feed_dict={
            image_tensor:
            np.expand_dims(preprocessed_input.astype('float32') / 255.0,
                           axis=0),
            sample_weights:
            np.array([1]),
            labels_tensor:
            np.reshape(one_hot, (-1, 3))
        })
    print(np.reshape(prediction_matrix, (-1, 3)))
    print(preprocessed_input.shape)
    print(target_conv_layer_value.shape)
    print(target_conv_layer_grad_value.shape)
    print(np.amax(target_conv_layer_value))
    print(np.amin(target_conv_layer_value))
    print(np.amax(target_conv_layer_grad_value))
    print(np.amin(target_conv_layer_grad_value))
utils.visualize(preprocessed_input, target_conv_layer_value[0],
                target_conv_layer_grad_value[0])
# cam, heatmap = grad_cam(graph, preprocessed_input, predicted_,
#                         "block5_conv3")
# cv2.imwrite("gradcam.jpg", cam)
