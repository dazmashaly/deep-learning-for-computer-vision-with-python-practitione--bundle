from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.core import Dropout
import tensorflow as tf

class AlexNet:
    @staticmethod
    def bulid (width,height,depth,classes,reg = 0.0002):
        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1

        if K.image_data_format() =="channels_first":
            inputShape = (depth,height,width)
            chanDim = 1

        
        #block 1 
        model.add(Conv2D(96,(11,11),strides=(4,4),input_shape = inputShape,padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Dropout(0.25))

        #block 2 
        model.add(Conv2D(256,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Dropout(0.25))
        
        #block 3
        model.add(Conv2D(384,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(384,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Dropout(0.25))
        
        #block 4
        model.add(Flatten())

        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model