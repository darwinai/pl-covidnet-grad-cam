from keras.applications.vgg16 import (VGG16, preprocess_input,
                                      decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
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


def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [
        grad if grad is not None else tf.zeros_like(var)
        for var, grad in zip(var_list, grads)
    ]


def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes
                                                  )
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(
        input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    model.summary()
    loss = K.sum(model.output)
    conv_output = [l for l in model.layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))

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


preprocessed_input = load_image(sys.argv[1])

model = VGG16(weights='imagenet')

predictions = model.predict(preprocessed_input)
print(predictions)
top_1 = decode_predictions(predictions)[0][0]
print(decode_predictions(predictions))
print('Predicted class:')
print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

predicted_class = np.argmax(predictions)
print(f"predicted_class: {predicted_class}")
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class,
                        "block5_conv3")
cv2.imwrite("gradcam.jpg", cam)