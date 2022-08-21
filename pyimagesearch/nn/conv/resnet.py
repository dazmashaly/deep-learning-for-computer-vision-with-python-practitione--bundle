from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.regularizers import L2
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers import add
from numpy import arange, pad

class Resnet:
    def residual_module(data, k, strade, chanDim,red =False,reg = 0.0001,bnEps=2e-5,bnMom=0.9):

        #idenditity
        shortcut = data

        #block 1
        bn1 = BatchNormalization(axis =chanDim,epsilon=bnEps,momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(k*0.25),(1,1),use_bias=False,kernel_regularizer=L2(reg))(act1)

        #block 2
        bn2 = BatchNormalization(axis =chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(k*0.25),(3,3),strides=strade,padding="same",use_bias=False,kernel_regularizer=L2(reg))(act2)

        #block 3
        bn3 = BatchNormalization(axis =chanDim,epsilon=bnEps,momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(k,(1,1),use_bias=False,kernel_regularizer=L2(reg))(act3)

        #reduce
        if red :
            shortcut = Conv2D(k,(1,1),strides=strade,use_bias=False,kernel_regularizer=L2(reg))(act1)

        x = add([conv3,shortcut])
        return x

    def build(width,height,depth,classes,stages,filters,reg=0.0001,bnEps=2e-5,bnMom=0.9,dataset="cifar"):

        inputShape = (height,width,depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        
        inputs = Input(shape = inputShape)
        x = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(inputs)

        if dataset == "cifar":
            x = Conv2D(filters[0],(3,3),use_bias=False,padding="same",kernel_regularizer=L2(reg))(x)
        
        for i in range(0,len(stages)):
            strade = (1,1) if i == 0 else (2,2)
            x = Resnet.residual_module(x,filters[i+1],strade,chanDim,red=True,bnEps=bnEps,bnMom=bnMom)
            for j in range(0,stages[i]-1):
                x = Resnet.residual_module(x,filters[i+1],(1,1),chanDim,red=True,bnEps=bnEps,bnMom=bnMom) 
        
        x = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8,8),name="pool5")(x)
        

        x = Flatten(name="flatten")(x)
        x = Dense(classes,kernel_regularizer=L2(reg))(x)
        x = Activation("softmax",name="softmax")(x)

        model = Model(inputs,x,name="resnet")

        return model