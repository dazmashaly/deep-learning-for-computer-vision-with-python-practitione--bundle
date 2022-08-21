import argparse
import matplotlib
matplotlib.use("Agg")

from config import tiny_img_cofg as config
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
import keras.backend as K
from pyimagesearch.preprocessing.meanPreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datagenerateor import Hdf5DatasetGenerator
from keras.models import load_model
from pyimagesearch.nn.conv.deepgnet import DeeperGoogLeNet
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c","--cheakpoints",required=True)
ap.add_argument("-m","--model",type=str)
ap.add_argument("-s","--startep",type=int)
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=35,height_shift_range=0.2,width_shift_range=0.2,zoom_range=0.15,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
means = json.loads(open(config.DATA_MEAN).read())

sp = SimplePreprocessor(64,64)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
iap = ImageToArrayPreprocessor()

trainGen = Hdf5DatasetGenerator(config.TRAIN_HDF5,64,aug=aug,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
valGen = Hdf5DatasetGenerator(config.VAL_HDF5,64,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)

if args["model"] is None:
    print("[INFO] compiling model...")
    model = DeeperGoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES,reg=0.002)
    opt = tf.keras.optimizers.SGD(learning_rate=0.05,decay = 0.01/70,momentum = 1,nesterov = True)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

else:
    print("[INFO] loading {}....".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate : {}...".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr,0.02)
    print("[INFO] new learning rate : {}...".format(K.get_value(model.optimizer.lr)))

callbacks = [TrainingMonitor(config.FIG_PATH,jsonPath=config.json_path,startAt=args["startep"])]
print("[INFO] training....")

model.fit_generator(
trainGen.generator(),
steps_per_epoch=trainGen.numImages // 64,
validation_data=valGen.generator(),
validation_steps=valGen.numImages // 64,
epochs=30,
max_queue_size=64 * 2,
callbacks=callbacks, verbose=1)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH,overwrite=True)
trainGen.close()
valGen.close()
