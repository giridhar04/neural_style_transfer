import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras import models 
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.models import load_model


def load_img(path):
    max_dim = 512
    img = Image.open(path)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(img, axis=0)
    return img

def load_and_process_img(path):
    img = load_img(path)
    img = tf.keras.applications.vgg19.preprocess_input(img)

    # print("Image type is: ", type(img))
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
        
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x