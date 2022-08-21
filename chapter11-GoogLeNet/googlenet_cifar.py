import matplotlib
matplotlib.use("agg")
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.nn.conv.minigooglenet import MiniGoogLeNet
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from pyimagesearch.callbacks import trainingmonitor

num_epochs = 70
init_lr = 0.015

def poly_decay(epoch):
    maxEpochs = num_epochs
    baselr = init_lr
    power = 0.5
    LR = baselr *(1-(epoch/float(maxEpochs))) ** power
    return LR

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to save")
ap.add_argument("-m","--model",required=True,help="path to save")
args = vars(ap.parse_args())

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

figPath = os.path.sep.join([args["output"],"{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"],"{}.json".format(os.getpid())])
callbacks = [trainingmonitor.TrainingMonitor(figPath,jsonPath),LearningRateScheduler(poly_decay)]


print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate =init_lr, momentum = 0.9, nesterov=True)
model = MiniGoogLeNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training the model...")
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),steps_per_epoch=len(x_train) //64,epochs=num_epochs,callbacks=callbacks,batch_size=40,verbose=1)
print("[INFO] evaluating the model...")
preds = model.predict(x_test,batch_size=64)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=label_names))

#plot the loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,70),H.history["loss"],label="training loss")
plt.plot(np.arange(0,70),H.history["val_loss"],label="validation loss")
plt.plot(np.arange(0,70),H.history["accuracy"],label="training acc")
plt.plot(np.arange(0,70),H.history["val_accuracy"],label="validation acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()