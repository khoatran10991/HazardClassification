#!/usr/bin/env python3

import numpy as np
import os
import cv2
from keras.applications import imagenet_utils
import pickle
from train_model import build_model

class HazardClassification():
    def __init__(self, dir='./'):
        self.label_map = pickle.load(open(os.path.join(dir, 'label_map.pkl'), 'rb'))
        self.num_class = len(self.label_map)
        self.label_key = list(self.label_map.keys())
        self.label_value = list(self.label_map.values())

        _, self.model = build_model(self.num_class)
        self.model.load_weights(os.path.join(dir, 'ckpt/model_best_ckpt.h5'))

    def run(self, image, return_label=True):
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(image, 0)
        image = imagenet_utils.preprocess_input(image)

        result = self.model.predict(image)
        result = np.argmax(result, axis=1)

        if return_label:
            return self.decode_label(result)
        else:
            return result

    def decode_label(self, result):
        return self.label_key[self.label_value.index(result)]    
