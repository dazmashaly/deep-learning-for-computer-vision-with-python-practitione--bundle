from json import load
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import os
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-n","--models",required=True,help="path to input ")
args = vars(ap.parse_args())
((x_train,y_train),(x_test,y_test)) = cifar10.load_data()
x_train = x_train.astype("float") /255.0
x_test = x_test.astype("float") /255.0
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
modelPaths = os.path.sep.join([args["models"],"*.model"])
modelPaths = list(glob.glob(modelPaths))
models =[]
for(i,modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i+1,len(modelPaths)))
    models.append(load_model(modelPath))

print("[INFO] evaluating ensemble..")
preds = []
for model in models:
    preds.append(model.predict(x_test,batch_size = 64))

preds = np.average(preds,axis=0)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=labelNames))