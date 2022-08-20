import tensorflow as tf
import time

from tensorflow.keras import models 
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.models import load_model


def get_content_loss(base_content, target):
    loss = tf.reduce_mean(tf.square(base_content - target))
    return loss


def gram_matrix(input_tensor):
    # We make the image channels first 
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    # Expects two images of dimension h, w, c
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    
    loss = tf.reduce_mean(tf.square(gram_style - gram_target))
    
    return loss