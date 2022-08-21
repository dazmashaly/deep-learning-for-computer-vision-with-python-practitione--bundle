from ast import Not
from unicodedata import name
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
from keras.regularizers import L2
from numpy import pad

class DeeperGoogLeNet:
    
    def conv_module(x, K, KX,KY,stride,chanDim,padding = "same",reg = 0.0005,name=None):
        #def a conv -> Bn => relu pattern
        (convName,bnName,actName) = (None,None,None)
        
        if name is not None:
            convName =name+"_conv"
            bnName = name+"_bn"
            actName =name+"_act"

        x = Conv2D(K,(KX,KY),strides=stride,padding=padding,kernel_regularizer=L2(reg),name=convName)(x)
        x = Activation("relu",name = actName)(x)
        x =BatchNormalization(axis=chanDim,name=bnName)(x)

        return x

    def inception_module(x,numK1x1,num3x3Reduce,num3x3,num5x5Reduce,num5x5,num1x1proj,stage,chanDim,reg=0.0005):
        #the first branch
        first = DeeperGoogLeNet.conv_module(x,numK1x1,1,1,(1,1),chanDim=chanDim,reg=reg,name=stage+"_first")

        #secoend branch
        second = DeeperGoogLeNet.conv_module(x,num3x3Reduce,1,1,(1,1),chanDim=chanDim,reg=reg,name=stage+"_second1")
        second = DeeperGoogLeNet.conv_module(second,num3x3,3,3,(1,1),chanDim=chanDim,reg=reg,name=stage+"_second2")

        #third branch
        third = DeeperGoogLeNet.conv_module(x,num5x5Reduce,1,1,(1,1),chanDim=chanDim,reg=reg,name=stage+"_third1")
        third = DeeperGoogLeNet.conv_module(third,num5x5,3,3,(1,1),chanDim=chanDim,reg=reg,name=stage+"_third2")

        #last branch
        foruth = MaxPooling2D((3,3),strides=(1,1),padding="same",name=stage+"_pool")(x)
        foruth = DeeperGoogLeNet.conv_module(foruth,num1x1proj,1,1,(1,1),chanDim,reg=reg,name=stage+"_foruth")

        x = concatenate([first,second,third,foruth],axis=chanDim,name=stage+"_mixed")

        return x

    def build(width,height,depth,classes,reg=0.0005):
        inputShape = (height,width,depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        
        inputs = Input(shape = inputShape)

        X = DeeperGoogLeNet.conv_module(inputs ,64,5,5,(1,1),chanDim=chanDim,reg=reg,name="block1")
        X = MaxPooling2D((3,3),strides=(2,2),padding="same",name="pool1")(X)
        X = DeeperGoogLeNet.conv_module(X ,64,1,1,(1,1),chanDim=chanDim,reg=reg,name="block2")
        X = DeeperGoogLeNet.conv_module(X,192,3,3,(1,1),chanDim=chanDim,reg=reg,name="block3")
        X = MaxPooling2D((3,3),strides=(2,2),padding="same",name="pool2")(X)

        X = DeeperGoogLeNet.inception_module(X,64,96,128,16,32,32,chanDim=chanDim,stage="3a",reg=reg)
        X = DeeperGoogLeNet.inception_module(X,128,128,192,32,96,64,chanDim=chanDim,stage="3b",reg=reg)
        X = MaxPooling2D((3,3),strides=(2,2),padding="same",name="pool3")(X)

        X = DeeperGoogLeNet.inception_module(X,192,96,208,16,48,64,chanDim=chanDim,stage="4a",reg=reg)
        X = DeeperGoogLeNet.inception_module(X,160,112,224,24,64,64,chanDim=chanDim,stage="4b",reg=reg)
        X = DeeperGoogLeNet.inception_module(X,128,128,256,24,64,64,chanDim=chanDim,stage="4c",reg=reg)
        X = DeeperGoogLeNet.inception_module(X,112,144,288,32,64,64,chanDim=chanDim,stage="4d",reg=reg)
        X = DeeperGoogLeNet.inception_module(X,256,160,320,32,128,128,chanDim=chanDim,stage="4e",reg=reg)
        X = MaxPooling2D((3,3),strides=(2,2),padding="same",name="pool4")(X)

        X = AveragePooling2D((4,4),name="pool5")(X)
        X = Dropout(0.4,name="do")(X)

        X = Flatten(name="flatten")(X)
        X = Dense(classes,kernel_regularizer=L2(reg),name="labels")(X)
        X = Activation("softmax",name="softmax")(X)

        model = Model(inputs,X,name="googlenet")

        return model