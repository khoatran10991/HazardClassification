#!/usr/bin/env python3

import numpy as np
import os
import cv2
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
import pickle
from keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from efficientnet.keras import EfficientNetB2
from keras.models import Model

class HazardClassification():
    def __init__(self, dir='./'):
        self.label_map = pickle.load(open(os.path.join(dir, 'label_map.pkl'), 'rb'))
        self.num_class = len(self.label_map)
        self.label_key = list(self.label_map.keys())
        self.label_value = list(self.label_map.values())

        _, self.model = self.build_model(self.num_class)
        self.model.load_weights(os.path.join(dir, 'ckpt/model_best_ckpt.h5'))
        print("LABEL", self.label_map)

    def run(self, image, return_label=True):
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
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

    def build_model(self, num_class):
        print("BUILDING MODEL TESTING...")
        # Load model EfficientNetB2 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
        baseModel = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

        # Xây thêm các layer
        # Lấy output của ConvNet trong EfficientNetB2
        fcHead = baseModel.output

        # Flatten trước khi dùng FCs
        fcHead = Flatten()(fcHead)

        # Thêm FC
        fcHead = Dense(1024, activation='relu')(fcHead)
        fcHead = BatchNormalization()(fcHead)
        fcHead = Dropout(0.2)(fcHead)

        fcHead = Dense(512, activation='relu')(fcHead)
        fcHead = BatchNormalization()(fcHead)
        fcHead = Dropout(0.2)(fcHead)

        fcHead = Dense(256, activation='relu')(fcHead)
        fcHead = BatchNormalization()(fcHead)
        # Output layer với softmax activation
        fcHead = Dense(num_class, activation='softmax')(fcHead)

        # Xây dựng model bằng việc nối ConvNet của EfficientNetB2 và fcHead
        model = Model(inputs=baseModel.input, outputs=fcHead)
        return baseModel, model
