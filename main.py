#!/usr/bin/env python3
import os
import sys
import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from tensorflow.python import keras
from keras.preprocessing.image import load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras import backend as K

from argparser import parse_args


def main():
    """ Mostly inspired from 
        https://github.com/totti0223/gradcamplusplus/blob/master/gradcamutils.py#L10 """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Makes TensorFlow shut up
    args = parse_args()
    logger = get_logger(args.verbose)

    logger.info("Loading the input image and resizing it to (224, 224) as required by the model.")
    image = np.array(load_img(args.input, target_size=(224, 224)), dtype=np.uint8)

    logger.info("Pre-processing the image like ImageNet's were when VGG was trained (sub mean from channels) "
                "also add the batch dimension (only one image)")
    image_processed = preprocess_input(np.expand_dims(image, axis=0))

    logger.info("Instantiate the pre-trained VGG19")
    model = VGG19(include_top=True, input_shape=(224, 224, 3))

    logger.info("Gets which class is the most activated")
    prediction = model.predict(image_processed)
    predicted_class = np.argmax(prediction)
    predicted_class_name = decode_predictions(prediction, top=1)[0][0][1]

    logger.info("Gets the tensor object (a scalar here) which is activated when showing the image")
    y_c = model.output[0, predicted_class]

    logger.info("Gets the tensor object corresponding to the output of the studied convolution layer")
    A_tensor = model.get_layer(args.layer).output

    logger.info("Gets the tensor containing the gradient of y_c w.r.t. A")
    gradient_tensor = K.gradients(y_c, A_tensor)[0]

    logger.info("Creates a function that takes as input the model's input "
                "and outputs the convolutions' result and the gradient of the prediction w.r.t. it")
    run_graph = K.function([model.input], [A_tensor, gradient_tensor])

    logger.info("Runs the graph using the inputted image")
    A, gradient = run_graph([image_processed])
    A, gradient = A[0], gradient[0]  # Gets the result of the first batch dimension

    logger.info("Performs global average pooling onto the activation gradient")
    alpha_c = np.mean(gradient, axis=(0, 1))

    logger.info("Weighs the filters maps with the activation coefficient")
    L_c = np.dot(A, alpha_c)

    logger.info("Resizes the localisation map to match the input image's size")
    L_c = zoom(L_c, 224/L_c.shape[0])

    logger.info("Plots the original image and the superimposed heat map")
    plt.subplots(nrows=1, ncols=2, dpi=160, figsize=(7, 4))
    plt.subplots_adjust(left=0.01, bottom=0.0, right=0.99, top=0.96, wspace=0.11, hspace=0.2)
    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(122)
    plt.title("{}th dimension ({}) \nw.r.t layer {}".format(predicted_class, predicted_class_name, args.layer))
    plt.imshow(image)
    plt.imshow(L_c, alpha=0.5, cmap="jet")
    plt.axis("off")

    if args.output is not None:
        logger.info("Saves the figure under {}".format(args.output))
        plt.savefig(args.output, dpi=300)

    if not args.quiet:
        plt.show()


def get_logger(verbose: bool) -> logging.Logger:
    """ Initialises the logger object """
    logger = logging.getLogger(name="Root")
    logger.setLevel(logging.INFO)

    # Only enable the logger's output if verbose is on
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info("Verbose enabled")

    return logger


if __name__ == "__main__":
    main()
