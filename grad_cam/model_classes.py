import tensorflow as tf
import numpy as np
import os

from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, args):
        self.args = args
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.load_model()

    def load_model(self):
        saver = tf.train.import_meta_graph(
            os.path.join(self.args.weightspath, self.args.metaname))
        saver.restore(self.sess,
                      os.path.join(self.args.weightspath, self.args.ckptname))

    def get_session(self):
        return self.sess

    def get_conv_output_tensor(self):
        graph = tf.get_default_graph()
        return graph.get_tensor_by_name(self.args.last_conv_out_tensor)

    @abstractmethod
    def get_loss_op(self):
        pass

    @abstractmethod
    def get_feed_dict(self, preprocessed_input, one_hot):
        pass

    @abstractmethod
    def get_one_hot(self, prediction_matrix):
        pass

    def __del__(self):
        self.sess.close()


class Covidnet(Model):
    def __init__(self, args):
        super().__init__(args)

    def get_covidnet_loss(self, pred_logit_tensor, labels_tensor,
                          sample_weights):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=pred_logit_tensor, labels=labels_tensor) *
            sample_weights)

    def get_loss_op(self):
        graph = tf.get_default_graph()
        labels_tensor = graph.get_tensor_by_name(self.args.labels_tensor)
        sample_weights = graph.get_tensor_by_name(
            self.args.sample_weights_tensor)
        pred_logit_tensor = graph.get_tensor_by_name(
            self.args.out_logit_tensor)
        return self.get_covidnet_loss(pred_logit_tensor, labels_tensor,
                                      sample_weights)

    def get_feed_dict(self, preprocessed_input, one_hot):
        graph = tf.get_default_graph()

        image_tensor = graph.get_tensor_by_name(self.args.in_tensor)
        labels_tensor = graph.get_tensor_by_name(self.args.labels_tensor)
        sample_weights = graph.get_tensor_by_name(
            self.args.sample_weights_tensor)

        return {
            image_tensor:
            np.expand_dims(preprocessed_input.astype('float32') / 255.0,
                           axis=0),
            sample_weights:
            np.array([1]),
            labels_tensor:
            np.reshape(one_hot, (-1, 3))
        }

    def get_one_hot(self, prediction_matrix):
        one_hot = np.zeros_like(prediction_matrix[0])
        one_hot[np.argmax(prediction_matrix[0])] = 1
        return one_hot
