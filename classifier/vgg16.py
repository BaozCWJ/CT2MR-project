import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is not  None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1',allow_pickle=True).item()
            print("npy file loaded")
        else:
            self.data_dict = None
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())



    def build(self,rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # # Convert RGB to BGR
        # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [64, 64, 1]
        # assert green.get_shape().as_list()[1:] == [64, 64, 1]
        # assert blue.get_shape().as_list()[1:] == [64, 64, 1]
        # bgr = tf.concat(axis=3, values=[
        #     blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #     red - VGG_MEAN[2],
        # ])
        # assert bgr.get_shape().as_list()[1:] == [64, 64, 3]

        self.conv1_1 = self.conv_layer(rgb_scaled,3,1,16,"conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1,3,16,16, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1,3,16,32, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1,3,32,32, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2,3,32,64, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1,3,64,64, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2,3,64,64, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3,3,64,128, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1,3,128,128, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2,3,128,128, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4,3,128,128, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1,3,128,128, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2,3,128,128, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5,1024, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6,250, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7,2, "fc8")

        prob = tf.nn.softmax(self.fc8, name="prob")
        logit = self.fc8

        print(("build model finished: %ds" % (time.time() - start_time)))
        return prob,logit

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom,kernel_size, in_channels,out_channels,name):
        with tf.variable_scope(name):
            conv_weights = tf.get_variable(name+"_weight",[kernel_size,kernel_size,in_channels,out_channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv_biases = tf.get_variable(name+"_bias", [out_channels], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(bottom, conv_weights, [1, 1, 1, 1], padding='SAME')

            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom,out_channels, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = tf.get_variable(name+"_weight", [dim, out_channels],initializer=tf.truncated_normal_initializer(stddev=0.1))

            biases = tf.get_variable(name+"_bias", [out_channels], initializer=tf.constant_initializer(0.1))
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
