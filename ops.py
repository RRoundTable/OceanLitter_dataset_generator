import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias




def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    """
    :param input: input
    :param filter: filter 개수, channel
    :param kernel: kernel size
    :return: conv
    """
    with tf.name_scope(layer_name):
        network=tf.layers.conv2d(inputs=input,use_bias=False,filters=filter,kernel_size=kernel, strides=stride,padding="SAME")
        return network

def Gloval_Average_Pooing(x, stride=1):
    """
       width = np.shape(x)[1]
       height = np.shape(x)[2]
       pool_size = [width, height] # 한번에 pooling한다
       return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter 1*1*dim
       It is global average pooling without tflearn
       """
    return global_avg_pool(x, name="Global_avg_pooling")


def Batch_Normalization(x,training,scope):
    """Stores the default arguments for the given set of list_ops."""
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):

        """
        condition에 따라 다른 함수 실행
        training == True:  accumulate the statistics of the moments into moving_mean and moving_variance 
        training == False : use the values of the moving_mean and the moving_variance.
        """
        print(type(x))
        training=tf.constant(training, dtype=tf.bool)
        return tf.cond(training, lambda : batch_norm(inputs=x, is_training=training, reuse=False),
                        lambda :batch_norm(inputs=x,is_training=training,reuse=True))


def Drop_out(x, rate,training):
    return tf.layers.dropout(inputs=x,rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Avergae_pooling(x,pool_size=[2,2],stride=2,padding="VALID"):
    return tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,padding=padding,strides=stride)


def Max_pooling(x,pool_size=[2,2],stride=2,padding="VALID"):
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Concatenation(layers):
    """
    axis 3 : (batch, channels, height, width)
    """
    return tf.concat(layers, axis=3)

def Linear(x):
    return tf.layers.dense(x,units=class_num, name="linear")