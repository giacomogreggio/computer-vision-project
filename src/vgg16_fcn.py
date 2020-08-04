import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPool2D, ZeroPadding2D, Cropping2D, Softmax, Add, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy

#Defining Imports
image_shape=(128,128,3)
n_classes=3
l2_value=5**-4
crop_value=((16,16),(16,16))


#Defining Base VGG architecture
input_layer = Input(shape=image_shape, name="input")
#VGG-block1
b1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", name="conv2d_b1_1")(input_layer)
b1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", name="conv2d_b1_2")(b1)
b1 = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool_b1")(b1)

#VGG-block2
b2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b2_1")(b1)
b2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b2_2")(b2)
b2 = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool_b2")(b2)

#VGG-block3
b3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b3_1")(b2)
b3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b3_2")(b3)
b3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b3_3")(b3)
b3 = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool_b3")(b3)

#VGG-block4
b4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b4_1")(b3)
b4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b4_2")(b4)
b4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b4_3")(b4)
b4 = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool_b4")(b4)

#VGG-block5
b5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b5_1")(b4)
b5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b5_2")(b5)
b5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="conv2d_b5_3")(b5)
b5 = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool_b5")(b5)

#Importing and locking weights
vgg_model = Model(input_layer, b5)
vgg16= VGG16(weights="imagenet", include_top=False)
vgg16.save_weights("./weights.h5")
vgg_model.load_weights("./weights.h5")
vgg_model.trainable=False
vgg_model.summary()



#FCN-32
fcn_32_block = Conv2D(4096, kernel_size=(7,7), activation='relu', kernel_regularizer=l2(l2_value), padding="same", name="conv2d_fcn32_1")(b5)
fcn_32_block = Dropout(0.5)(fcn_32_block)
fcn_32_block = Conv2D(4096, kernel_size=(1,1), activation='relu', kernel_regularizer=l2(l2_value), padding="same", name="conv2d_fcn32_2")(fcn_32_block)
fcn_32_block = Dropout(0.5)(fcn_32_block)
fcn_32_block = Conv2D(n_classes, kernel_size=(1,1), kernel_regularizer=l2(l2_value), padding="same", name="conv2d_fcn32_3")(fcn_32_block)

fcn_32_transpose = Conv2DTranspose(n_classes, kernel_size=(64,64), strides=(32,32), name="conv2dtrans_fcn32")(fcn_32_block)
fcn_32_crop = Cropping2D(crop_value, name="crop_fcn32")(fcn_32_transpose)
fcn_32_crop=Softmax(axis=-1)(fcn_32_crop)
fcn_32 = Model(input_layer, fcn_32_crop)
fcn_32.summary()



#FCN-16
fcn_16_block_fcn32 = Conv2DTranspose(n_classes, activation='softmax', kernel_size=(1,1), padding="valid", strides=(1,1), name="fcn_16_block_fcn32")(fcn_32_block)

fcn_16_block = Conv2D(n_classes, kernel_size=(1,1), activation='relu', kernel_regularizer=l2(l2_value), padding="valid", name="conv2d_fcn16_1")(b4)
fcn_16_block = Cropping2D(((2,2,),(2,2)))(fcn_16_block)
fcn_16_block = Add()([fcn_16_block_fcn32,fcn_16_block])
fcn_16_block = Conv2DTranspose(n_classes, activation='softmax', kernel_size=(32,32), strides=(16,16))(fcn_16_block)
fcn_16 = Model(input_layer, fcn_16_block)

tf.keras.utils.plot_model(fcn_16, show_shapes=True)
