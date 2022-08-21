from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.minivggnet import MiniVGGnet
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input ")
args = vars(ap.parse_args())

print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(64,64)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap,iap])
data,labels = sdl.load(imagePaths,verbose=500)
data = data.astype("float")/255.0
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.25)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

#aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

print("[compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate=0.05)
model = MiniVGGnet.build(width=64,  height=64,depth=3,classes=len(classNames))
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
print("[INFO] training the network...")
#H = model.fit_generator(aug.flow(x_train,y_train,batch_size=32),validation_data=(x_test,y_test),steps_per_epoch=len(x_train)/32,epochs=100,verbose=1)
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=32, epochs=100, verbose=1)

print("[info] evaluating network...")
preds = model.predict(x_test,batch_size=32)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=classNames))

#plot the loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100),H.history["loss"],label="training loss")
plt.plot(np.arange(0,100),H.history["val_loss"],label="validation loss")
plt.plot(np.arange(0,100),H.history["accuracy"],label="training acc")
plt.plot(np.arange(0,100),H.history["val_accuracy"],label="validation acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()

