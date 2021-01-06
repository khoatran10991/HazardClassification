#!/usr/bin/env python3

from keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization
from efficientnet.keras import EfficientNetB2
from keras.models import Model

def build_model(num_class):
    print("BUILDING MODEL...")
    # Load model EfficientNetB2 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
    baseModel = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Xây thêm các layer
    # Lấy output của ConvNet trong EfficientNetB2
    fcHead = baseModel.output

    # Flatten trước khi dùng FCs
    fcHead = Flatten()(fcHead)

    # Thêm FC
    fcHead = Dense(2048, activation='relu')(fcHead)
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
