import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, MaxPool2D, ZeroPadding2D, Cropping2D, Softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy


def create_vgg_fcn_32(n_classes, image_shape, crop_value,l2_value=5**-4):
    l2_value = 5 ** -4
    
    input_layer = Input(shape=image_shape)
    x = ZeroPadding2D(100)(input_layer)
    x = Conv2D(filters=64,kernel_size=(3,3),padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=64,kernel_size=(3,3),padding="valid", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="valid", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    model = Model(input_layer, x)
    vgg16= VGG16(weights="imagenet", include_top=False)
    vgg16.save_weights("./weights.h5")
    model.load_weights("./weights.h5", by_name=True)
    model.trainable = False

    x = Conv2D(4096, kernel_size=(7,7), activation='relu', kernel_regularizer=l2(l2_value), padding="valid", kernel_initializer=tf.keras.initializers.Zeros())(x)
    x = Conv2D(4096, kernel_size=(1,1), activation='relu', kernel_regularizer=l2(l2_value), padding="valid", kernel_initializer=tf.keras.initializers.Zeros())(x)
    x = Conv2D(n_classes, kernel_size=(1,1), activation='relu', kernel_regularizer=l2(l2_value), padding="valid", kernel_initializer=tf.keras.initializers.Zeros())(x)
    x = Conv2DTranspose(n_classes, activation='softmax', kernel_size=(64,64), strides=(32,32), kernel_regularizer=l2(l2_value), use_bias=False, kernel_initializer=tf.keras.initializers.Zeros())(x)

    x = Cropping2D(crop_value)(x)
    model = Model(input_layer, x)
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model