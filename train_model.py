#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import csv
import pickle
import cv2
from sklearn.model_selection import train_test_split
from efficientnet.keras import EfficientNetB2
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten
from generator import DataGenerator
from datetime import datetime


def load_data(path_image):
    print("LOADING IMAGES...")

    #Load image from path_image
    image_path = os.listdir(path_image)
    image_path = list(filter(lambda x: x[-3:].lower() == 'jpg' or x[-3:].lower() == 'png', image_path))
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

def encode_label(list_label, save_file):
    print("ENCODING LABELS...")
    dir = './'
    if os.path.exists(os.path.join(dir, save_file)):
        print("LOADING LABEL MAP")
        label_map = pickle.load(open(os.path.join(dir, save_file), 'rb'))
    else:
        print("SAVE LABEL MAP")
        set_list_label = set(list_label)
        set_list_label = sorted(set_list_label)
        label_map = dict((c, i) for i, c in enumerate(set_list_label))
        pickle.dump(label_map, open(os.path.join(dir, save_file), 'wb'))

    print("LABEL MAP", label_map)   
    encoded = [label_map[x] for x in list_label]
    encoded = to_categorical(encoded)
    print("Load or Save file %s success" % save_file)
    return encoded


def build_model(num_class):
    print("BUILDING MODEL...")
    # Load model EfficientNetB2 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
    baseModel = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Xây thêm các layer
    # Lấy output của ConvNet trong EfficientNetB2
    fcHead = baseModel.output

    # Flatten trước khi dùng FCs
    fcHead = Flatten()(fcHead)

    # Thêm FC
    fcHead = Dense(512, activation='relu')(fcHead)
    fcHead = Dropout(0.2)(fcHead)
    fcHead = Dense(256, activation='relu')(fcHead)

    # Output layer với softmax activation
    fcHead = Dense(num_class, activation='softmax')(fcHead)

    # Xây dựng model bằng việc nối ConvNet của EfficientNetB2 và fcHead
    model = Model(inputs=baseModel.input, outputs=fcHead)
    return baseModel, model


def train_model(model, baseModel, X_train, y_train, X_test=None, y_test=None, args=None, n_classes=0, batch_size=32, ckpt_path='./ckpt'):
    """
    TRAIN MODEL HAZARD DETECT
    """
    aug_train = DataGenerator(X_train, y_train, args.img_path, 3, batch_size=batch_size, n_classes=n_classes)
    checkpoint = ModelCheckpoint(os.path.join(ckpt_path, 'model_best_ckpt.h5'), monitor="val_loss",
                                 save_best_only=True, mode='min', save_weights_only=True, save_freq='epoch')
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    # Load checkpoint
    if os.path.exists(os.path.join(ckpt_path, 'model_best_ckpt.h5')):
        print("LOADING MODEL WEIGHT...")
        model.load_weights(os.path.join(ckpt_path, 'model_best_ckpt.h5'))
    else:
        print("CREATE MODEL WEIGHT FILE...")
    
    if(args.step != 2):
        print("TRAINING MODEL STEP 1...")
        # freeze EfficientNetB2 model
        for layer in baseModel.layers:
            layer.trainable = False
        opt = RMSprop(0.001)
        model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        if (args.validation):
            aug_test = DataGenerator(X_test, y_test, args.img_path, 3, batch_size=batch_size, n_classes=n_classes)
            H = model.fit(aug_train, validation_data=aug_test, epochs=args.epoch_step_1, callbacks=[checkpoint, early_stop])
        else:
            H = model.fit(aug_train, epochs=args.epoch_step_1, callbacks=[checkpoint, early_stop])

    if(args.step != 1):
        print("TRAINING MODEL STEP 2...")
        # unfreeze all CNN layer in EfficientNetB2:
        for layer in baseModel.layers:
            layer.trainable = True

        opt = Adam(lr=0.001, decay=5e-5)
        model.compile(opt, 'categorical_crossentropy', ['accuracy'])
        if (args.validation):
            aug_test = DataGenerator(X_test, y_test, args.img_path, 3, batch_size=batch_size, n_classes=n_classes)
            H = model.fit(aug_train, validation_data=aug_test, epochs=args.epoch_step_2, callbacks=[checkpoint, early_stop])
        else:
            H = model.fit(aug_train, epochs=args.epoch_step_2, callbacks=[checkpoint, early_stop])
    
    print("FINISH TRAINING MODEL...")

def main(args):
    print("START MAIN CLASS TRAINING MODEL")
    list_image, list_label = load_data(args.img_path)
    labels = encode_label(list_label, args.mapping_file)
    n_classes = len(set(list_label))
    print("NUM CLASSES", n_classes)
    print("LIST CLASSES AFTER SHUFFLE", set(list_label))
    baseModel, mainModel = build_model(n_classes)
    
    if (args.validation):
        X_train, X_test, y_train, y_test = train_test_split(list_image, labels, test_size=0.2, random_state=42)
        train_model(model=mainModel, baseModel=baseModel, X_train=X_train, X_test=X_test,y_train=y_train, y_test=y_test, args=args, n_classes=n_classes)
    else:
        train_model(model=mainModel, baseModel=baseModel, X_train=list_image,y_train=labels, args=args, n_classes=n_classes)
    print("FINISH MAIN CLASS TRAINING MODEL")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='Path to folder which contains images.', type=str, default='./images-cropped')
    parser.add_argument('--mapping_file', help='Path to save label map file.', type=str, default='label_map.pkl')
    parser.add_argument('--epoch_step_1', help='Number of epochs for training step 1.', type=int, default=30)
    parser.add_argument('--epoch_step_2', help='Number of epochs for training step 2.', type=int, default=100)
    parser.add_argument('--validation', help='Wheather to split data for validation.', type=bool, default=True)
    parser.add_argument('--step', help='Training model step (1 or 2)', type=int, default=3)

    args = parser.parse_args()
    main(args)
