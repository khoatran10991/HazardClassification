#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import csv
import pickle
import cv2


def load_data(path_image):
    print("LOADING IMAGES...")

    #Load image from path_image
    image_path = os.listdir(path_image)
    image_path = list(filter(
        lambda x: x[-3:].lower() == 'jpg' or x[-3:].lower() == 'png', image_path))
    #image_path = np.repeat(image_path, 10)
    random.shuffle(image_path)

    #Result variable
    list_image = []
    list_label = []

    #Mapping image with label
    for (j, imagePath) in enumerate(image_path):
        listPath = imagePath.split('-')
        list_image.append(imagePath)
        list_label.append(listPath[0])

    num_img = len(list_image)
    print("Total images: %d" % num_img)
    return list_image, list_label

def main(args):
    print("START MAIN CLASS TRAINING MODEL")
    list_image, list_label = load_data(args.img_path)
    n_classes = len(set(list_label))
    print("NUM CLASSES", n_classes)
    print("LIST CLASSES", set(list_label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='Path to folder which contains images.',
                        type=str, default='./images-cropped')

    args = parser.parse_args()
    main(args)
