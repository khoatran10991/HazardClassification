#!/usr/bin/env python3

import sys
import cv2
import os
from keras.callbacks import Callback
from process import HazardClassification

class TestModel(Callback):
    """
    Test model after epoch
    """

    def on_epoch_end(self, epoch, logs=None, path_test="./images-test"):
        print("\nTESTING WITH WEIGHT UPDATED....")
        classifier = HazardClassification()

        # Loading image"
        fileNames = list(filter(lambda jpg: jpg[-3:].lower() == 'jpg', os.listdir(path_test)))
        correct = 0
        miss = 0
        for file in fileNames:
            img = cv2.imread(os.path.join(path_test, file), cv2.IMREAD_UNCHANGED)
            imgPath = file.split('-')
            class_original = imgPath[0]
            class_res = classifier.run(image=img)
            if(class_original == class_res):
                correct = correct + 1
            else:
                miss = miss + 1
        total = correct + miss
        correct = correct / total
        miss = miss / total
        print("CORRECT %f - MISS %f \n" %(correct, miss))
