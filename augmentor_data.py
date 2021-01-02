import os
import csv
import cv2
import Augmentor

def augmentor(path_in, path_out):
    pipe = Augmentor.Pipeline(path_in, path_out)
    pipe.random_distortion(probability=0.7, grid_width=4,grid_height=4, magnitude=8)
    pipe.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    pipe.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
    pipe.skew(probability=0.7)
    pipe.crop_random(probability=0.3, percentage_area=0.9)
    pipe.shear(probability=0.7, max_shear_left=5, max_shear_right=5)
    pipe.sample(3000)
    return

def main(args=None):
    """
    Main function Agumentor Data
    """
    augmentor("images-cropped", 'images-agumentor')


if __name__ == "__main__":
    main()
