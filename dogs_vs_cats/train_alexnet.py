import matplotlib
matplotlib.use("Agg")

from config import d_v_c_config as config
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import learning_curve, train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.patchpreprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.meanPreprocessor import MeanPreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datagenerateor import Hdf5DatasetGenerator
from pyimagesearch.nn.conv.alexnet import AlexNet
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os

aug = ImageDataGenerator(rotation_range=20,height_shift_range=0.2,width_shift_range=0.2,zoom_range=0.15,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227,227)
pp = PatchPreprocessor(227,227)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
iap = ImageToArrayPreprocessor()

trainGen = Hdf5DatasetGenerator(config.TRAIN_HDF5,128,aug=aug,preprocessors=[pp,mp,iap],classes=2)
valGen = Hdf5DatasetGenerator(config.VAL_HDF5,128,preprocessors=[sp,mp,iap],classes=2)

print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model = AlexNet.bulid(227,227,3,2,0.0002)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

#construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH,"{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

model.fit_generator(
trainGen.generator(),
steps_per_epoch=trainGen.numImages // 128,
validation_data=valGen.generator(),
validation_steps=valGen.numImages // 128,
epochs=75,
max_queue_size=128 ,
callbacks=callbacks, verbose=1)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH,overwrite=True)
trainGen.close()
valGen.close()
