import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.minivggnet import MiniVGGnet
import tensorflow as tf
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-n","--nmodels",type=int,default=5,help="path to input ")
ap.add_argument("-o","--output",required=True,help="path to output dirct ")
ap.add_argument("-m","--model",required=True,help="path to output models ")
args = vars(ap.parse_args())
((x_train,y_train),(x_test,y_test)) = cifar10.load_data()
x_train = x_train.astype("float") /255.0
x_test = x_test.astype("float") /255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
aug = ImageDataGenerator(rotation_range=30,height_shift_range=0.1,width_shift_range=0.1,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
for i in np.arange(0,args["nmodels"]):
    print("[INFO] training model {}/{}".format(i+1,args["nmodels"]))
    opt = tf.keras.optimizers.SGD(learning_rate=0.01,decay = 0.01/40,momentum = 0.9,nesterov = True)
    model = MiniVGGnet.build(width=32,height=32,depth=3,classes=10)
    model.compile(loss ="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    H = model.fit(aug.flow(x_train,y_train,batch_size=64),validation_data=(x_test,y_test),epochs=40,steps_per_epoch=len(x_train) //64,verbose=1)
    #save the model
    p = [args["model"],"model_{}.model".format(i)]
    model.save(os.path.sep.join(p))
    preds = model.predict(x_test,batch_size=64)
    #evaluate and save the report
    report = classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=labelNames)
    p = [args["output"],"model_{}.txt".format(i)]
    f = open(os.path.sep.join(p),"w")
    f.write(report)
    f.close()
    p = [args["output"],"model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,40),H.history["loss"],label="training loss")
    plt.plot(np.arange(0,40),H.history["val_loss"],label="validation loss")
    plt.plot(np.arange(0,40),H.history["accuracy"],label="training acc")
    plt.plot(np.arange(0,40),H.history["val_accuracy"],label="validation acc")
    plt.title("training loss and accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()

