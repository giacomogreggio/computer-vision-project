import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as back 
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input 
import os
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, MaxPool2D, ZeroPadding2D, Cropping2D, Softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
import h5py
from IPython.display import display
from PIL import Image
from skimage.io import imshow
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import trange, tqdm 
from time import sleep  