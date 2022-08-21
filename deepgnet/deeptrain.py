import argparse
from tkinter import E
import matplotlib
matplotlib.use("Agg")

from config import tiny_img_cofg as config
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
import keras.backend as K
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.callbacks.epochceackpoint import EpochCheckpoint
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from pyimagesearch.nn.conv.resnet import Resnet
import json
import tensorflow as tf
import sys 
import matplotlib.pyplot as plt
import numpy as np
import os

sys.setrecursionlimit(5000)

ap = argparse.ArgumentParser()
ap.add_argument("-c","--cheakpoints",required=True)
ap.add_argument("-m","--model",type=str)
ap.add_argument("-s","--startep",type=int)
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=35,height_shift_range=0.2,width_shift_range=0.2,zoom_range=0.15,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
print("[INFO] loading dataset...")
#replace fetch_mldata with 
((x_train,y_train),(x_test,y_test)) = cifar10.load_data()
x_train =x_train.astype("float")
x_test =x_test.astype("float")

mean = np.mean(x_train,axis=0)
x_train-=mean
x_test-=mean

#convert labels from int to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

label_names =["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
aug = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,fill_mode="nearest")


if args["model"] is None:
    print("[INFO] compiling model...")
    model = Resnet.build(32,32,3,10,(9,9,9),(64,64,128,256),reg=0.0003)
    opt = tf.keras.optimizers.SGD(learning_rate=0.1,momentum = 0.8,nesterov = True)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

else:
    print("[INFO] loading {}....".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate : {}...".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr,0.1)
    print("[INFO] new learning rate : {}...".format(K.get_value(model.optimizer.lr)))

callbacks = [EpochCheckpoint(config.MODEL_PATH,every=3,startAt=args["startep"]),TrainingMonitor(config.FIG_PATH,jsonPath=config.json_path,startAt=args["startep"])]
print("[INFO] training....")

model.fit_generator(aug.flow(x_train,y_train,batch_size=128),validation_data=(x_test,y_test),steps_per_epoch=len(x_train) // 128,epochs=20
,callbacks=callbacks,verbose=1)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH,overwrite=True)
