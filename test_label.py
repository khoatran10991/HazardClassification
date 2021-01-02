#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import os
from process import HazardClassification

def main():
    classifier = HazardClassification()

    # Loading image"
    fileName = list(filter(lambda jpg: jpg[-3:].lower() == 'jpg', os.listdir('./images-cropped/')))
    correct = 0
    miss = 0
    for file in fileName:
        img = cv2.imread(os.path.join("images-cropped", file), cv2.IMREAD_UNCHANGED)
        imgPath = file.split('-')
        class_original = imgPath[0]
        class_res = classifier.run(image=img)
        if(class_original == class_res):
            correct = correct + 1
        else:
            miss = miss + 1
            # print("File:", file)
            # print("Class_original:", class_original)
            # print("Class_res:", class_res)
            # print("Status: MISS")
        
    total = correct + miss
    correct = correct / total
    miss = miss / total
    print("CORRECT %f - MISS %f" %(correct, miss))

if __name__ == "__main__":
    main()
