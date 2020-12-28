#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import os
from process import HazardClassification

def main():
    classifier = HazardClassification()

    # Loading image"
    fileName = list(filter(lambda jpg: jpg[-3:].lower() == 'jpg', os.listdir('./test/')))
    for file in fileName:
        img = cv2.imread(os.path.join("test", file))

        class_res = classifier.run(img)
        img_out = cv2.resize(img, (round(
            img.shape[1]*0.3), round(img.shape[0]*0.3)), interpolation=cv2.INTER_AREA)
        cv2.imshow(class_res, img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
