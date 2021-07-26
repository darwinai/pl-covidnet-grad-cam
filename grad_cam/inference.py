import tensorflow as tf
import os

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from data import process_image_file

import numpy as np
from skimage.transform import resize
import cv2
import shutil


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
            image_tensor = graph.get_tensor_by_name(self.args.in_tensor)
            labels_tensor = graph.get_tensor_by_name(self.args.labels_tensor)
            sample_weights = graph.get_tensor_by_name(
                self.args.sample_weights_tensor)
            pred_logit_tensor = graph.get_tensor_by_name(
                self.args.out_logit_tensor)
            # Define loss
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
        # utils.visualize(preprocessed_input, target_conv_layer_value[0],
        #                 target_conv_layer_grad_value[0])
        self.generate_mask(preprocessed_input, target_conv_layer_value[0],
                           target_conv_layer_grad_value[0])

    def generate_mask(self, preprocessed_input, conv_output, conv_grad):

        weights = np.mean(conv_grad, axis=(0, 1))
        cam = np.zeros(conv_output.shape[0:2], dtype=np.float32)

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * conv_output[:, :, i]

        cam -= np.min(cam)
        cam = cam / np.max(cam)
        size = self.args.input_size
        # TO DO: resize heatmap to dimensions of raw input and mask the cropped out areas
        cam = resize(cam, (size, size), preserve_range=True)
        _, cam_heatmap = cv2.threshold(np.uint8(255 * cam), 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cam_heatmap = cam_heatmap / np.max(cam_heatmap)
        cam_heatmap = 255 - cam_heatmap
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_GRAY2RGB)
        purple = [105, 54, 169]
        for i in range(3):
            cam_heatmap[:, :, i] = cam_heatmap[:, :, i] / 255 * purple[i]

        file_name = self.args.imagefile.split('.')
        mask_file_name = f"{self.args.outputdir}/{file_name[0]}-mask.{file_name[1]}"
        print(f"Creating {mask_file_name} in {self.args.outputdir}")
        cv2.imwrite(mask_file_name, cam_heatmap)

        # TO DO: omit preprocessed input when mask fits the dimensions of the raw input
        preprocessed_input_file_path = f"{self.args.outputdir}/{file_name[0]}-preprocessed.{file_name[1]}"
        print(
            f"Creating {preprocessed_input_file_path} in {self.args.outputdir}"
        )
        cv2.imwrite(preprocessed_input_file_path, preprocessed_input)

        print(f"Copying over the input image to {self.args.outputdir}")
        shutil.copy(f"{self.args.inputdir}/{self.args.imagefile}",
                    self.args.outputdir)
