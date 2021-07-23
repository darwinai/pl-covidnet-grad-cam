import tensorflow as tf
import os

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from data import process_image_file
import utils

import numpy as np
import sys


def readInput():
    image = process_image_file(sys.argv[1], 0.08, 480)

    with open(sys.argv[2], 'r') as f:
        input = json.load(f)
        prediction_matrix = np.array([
            float(x)
            for x in [input['Normal'], input['Pneumonia'], input['COVID-19']]
        ])
    return image, prediction_matrix


def readInput2():
    image = process_image_file(sys.argv[1], 0.08, 480)

    with open(sys.argv[2], 'r') as f:
        input = json.load(f)
        print(input)
        prediction_matrix = np.array(input)
        print(prediction_matrix)
    return image, prediction_matrix


class Inference():
    '''
        the args dict should have:
        weightspath: str, metaname : str, ckptname: str
        '''
    def __init__(self, args):
        self.args = args

    def process_input(self):
        imagepath = f"{self.args.inputdir}/{self.args.imagefile}"
        image = process_image_file(imagepath, self.args.top_percent,
                                   self.args.input_size)
        pred_matrix = f"{self.args.inputdir}/{self.args.pred_matrix}"
        with open(pred_matrix, 'r') as f:
            input = json.load(f)
            print(input)
            prediction_matrix = np.array(input)
            print(prediction_matrix)
        return image, prediction_matrix

    def infer(self):
        preprocessed_input, prediction_matrix = self.process_input()

        tf.reset_default_graph()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            tf.get_default_graph()
            saver = tf.train.import_meta_graph(
                os.path.join(self.args.weightspath, self.args.metaname))
            saver.restore(
                sess, os.path.join(self.args.weightspath, self.args.ckptname))

            graph = tf.get_default_graph()
            # placeholders = [
            #     op for op in graph.get_operations() if op.type == "Placeholder"
            # ]
            # print(placeholders)
            image_tensor = graph.get_tensor_by_name(self.args.in_tensor)
            # pred_tensor = graph.get_tensor_by_name('norm_dense_1/Softmax:0')

            # x = preprocessed_input
            # x = x.astype('float32') / 255.0
            # pred = sess.run(
            #     pred_tensor,
            #     feed_dict={image_tensor: np.expand_dims(x, axis=0)})
            # print(pred)
            labels_tensor = graph.get_tensor_by_name(self.args.labels_tensor)
            sample_weights = graph.get_tensor_by_name(
                self.args.sample_weights_tensor)
            pred_logit_tensor = graph.get_tensor_by_name(
                self.args.out_logit_tensor)
            # pred_softmax_tensor = graph.get_tensor_by_name(
            #     self.args.out_softmax_tensor)
            # image_tensor = graph.get_tensor_by_name('input_1:0')
            # loss expects unscaled logits since it performs a softmax on logits internally for efficiency
            # Define loss and optimizer
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=pred_logit_tensor, labels=labels_tensor) *
                sample_weights)
            conv_output = graph.get_tensor_by_name(
                self.args.last_conv_out_tensor)
            target_conv_layer_grad = tf.gradients(loss_op, conv_output)[0]
            one_hot = np.zeros_like(prediction_matrix[0])
            one_hot[np.argmax(prediction_matrix[0])] = 1

            target_conv_layer_value, target_conv_layer_grad_value = sess.run(
                [conv_output, target_conv_layer_grad],
                feed_dict={
                    image_tensor:
                    np.expand_dims(preprocessed_input.astype('float32') /
                                   255.0,
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
