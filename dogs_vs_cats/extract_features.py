from keras.applications.resnet import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-o","--output",required=True,help= "path to output hdf5")
ap.add_argument("-b","--batchSize",type=int,default=16,help= "path to output hdf5")
ap.add_argument("-s","--bufferSize",type=int,default=1000,help= "path to output hdf5")
args = vars(ap.parse_args())
bs = args["batchSize"]


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)

dataset = HDF5DatasetWriter((len(imagePaths), 49*2048),args["output"], dataKey="features", bufSize=args["bufferSize"])
dataset.storeClassLabels(le.classes_)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()

for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []
    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 49*2048))
    dataset.add(features, batchLabels)
    pbar.update(i)
dataset.close()
pbar.finish()