from enum import unique
import imp
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.fcheadnet import FCHeadNet
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input ")
ap.add_argument("-m","--model",required=True,help="path to output ")
args = vars(ap.parse_args())

#initialize the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(224,224)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap,iap])
(data,labels) = sdl.load(imagePaths,verbose=500)
data = data.astype("float")/255.0

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.25,random_state=42)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

#network surgery
#load network without the head
baseModel = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))
#set up the new head
headModel = FCHeadNet.build(baseModel,len(classNames),256)
#place the head on top
model  = Model(inputs=baseModel.input,outputs = headModel )

#freeze the base
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO]compiling head...")
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
print("[INFO] training the network...")
#train the head for a while to warm up
model.fit_generator(aug.flow(x_train,y_train,batch_size=32),validation_data=(x_test,y_test),steps_per_epoch=len(x_train)/32,epochs=25,verbose=1)
print("[INFO] evaluating the network after warm up...")
preds = model.predict(x_test,batch_size=32)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=classNames))

#unfreeze some layers
for layer in baseModel.layers[15:]:
    layer.trainable = True

#recompile and train
print("[INFO]compiling head...")
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training the network...")
#train the head for a while to warm up
model.fit_generator(aug.flow(x_train,y_train,batch_size=32),validation_data=(x_test,y_test),steps_per_epoch=len(x_train)/32,epochs=100,verbose=1)
print("[INFO] evaluating the network after fine_tuning...")
preds = model.predict(x_test,batch_size=32)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=classNames))
print("[INFO]saving model...")
model.save(args["model"])