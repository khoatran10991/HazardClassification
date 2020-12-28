import argparse
import os
import csv
from PIL import Image, ImageFile

def croppedImage(path_in, path_out, label_file):
    """
    CROPPED IMAGE WITH POSITION IN LABEL FILE
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(label_file, newline='') as csvfile:
        label_csv = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(label_csv, None)
        line_count = 1
        file_count = 1
        for row in label_csv:
            print("STT %d : %s" %(line_count, row[0]))
            if os.path.exists(os.path.join(path_in, row[0])):
                img = Image.open(os.path.join(path_in, row[0]))
                xmin = int(float(row[1]))
                ymax = int(float(row[4]))
                xmax = int(float(row[3]))
                ymin = int(float(row[2]))
                print("STT %d xmin=%d xmax=%d ymin=%d ymax=%d" %
                      (file_count, xmin, xmax, ymin, ymax))
                img_res = img.crop((xmin, ymin, xmax, ymax))
                img_res.save(os.path.join(path_out, row[0]))
                file_count = file_count + 1
            line_count = line_count + 1

    return

def main(args):
    print("CROPPED IMAGE...")
    croppedImage(path_in=args.img_path_in,
                 path_out=args.img_path_out, label_file=args.label_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_in', help='Path to folder which contains images.', type=str, default='./images')
    parser.add_argument('--img_path_out', help='Path to folder which contains cropped images.', type=str, default='./cropped')
    parser.add_argument('--label_file', help='Path to label file.', type=str, default='Hazard-Classification-export.csv')

    args = parser.parse_args()
    main(args)

