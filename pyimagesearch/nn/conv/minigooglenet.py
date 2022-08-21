from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.core import Dropout

class MiniGoogLeNet:

    def conv_module(x, K, KX,KY,stride,chanDim,padding = "same"):
        #def a conv -> Bn => relu pattern
        x = Conv2D(K,(KX,KY),strides=stride,padding=padding)(x)
        x =BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)

        return x
    
    def inception_module(x,numK1x1,numK3x3,chanDim):
        conv_1x1 = MiniGoogLeNet.conv_module(x,numK1x1,1,1,(1,1),chanDim=chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x,numK3x3,3,3,(1,1),chanDim=chanDim)
        x = concatenate([conv_1x1,conv_3x3],axis=chanDim)

        return x
    
    def downsample_module(x,K,chanDim):
        conv_3x3 = MiniGoogLeNet.conv_module(x,K,3,3,(2,2),chanDim=chanDim,padding="valid")
        pool = MaxPooling2D((3,3),strides=(2,2))(x)
        x = concatenate([conv_3x3,pool],axis=chanDim)

        return x

    def build(width,height,depth,classes):
        inputShape = (height,width,depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        
        inputs = Input(shape = inputShape)
        X = MiniGoogLeNet.conv_module(inputs ,96,3,3,(1,1),chanDim=chanDim)
        X = MiniGoogLeNet.inception_module(X,32,32,chanDim)
        X = MiniGoogLeNet.inception_module(X,32,48,chanDim)
        X = MiniGoogLeNet.downsample_module(X,96,chanDim)

        X = MiniGoogLeNet.inception_module(X,112,48,chanDim)
        X = MiniGoogLeNet.inception_module(X,96,64,chanDim)
        X = MiniGoogLeNet.inception_module(X,80,80,chanDim)
        X = MiniGoogLeNet.inception_module(X,48,96,chanDim)
        X = MiniGoogLeNet.downsample_module(X,96,chanDim)

        X = MiniGoogLeNet.inception_module(X,176,160,chanDim)
        X = MiniGoogLeNet.inception_module(X,176,160,chanDim)
        X = AveragePooling2D((7,7))(X)
        X = Dropout(0.5)(X)

        X = Flatten()(X)
        X = Dense(classes)(X)
        X = Activation("softmax")(X)

        model = Model(inputs,X,name="googlenet")

        return model