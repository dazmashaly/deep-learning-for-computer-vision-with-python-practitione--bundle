from keras.applications.vgg16 import VGG16
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
ap.add_argument("-d","--dataset" ,required=True,help="path to input data")
ap.add_argument("-o","--output" ,required=True,help="path to HDF5 file")
ap.add_argument("-b","--batchsize" ,type = int,default= 32,help="batch size")
ap.add_argument("-s","--buffersize" ,type=int,default=1000,help="path to input data")
args=vars(ap.parse_args())

bs = args["batchsize"]
print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args['dataset']))
random.shuffle(imagePaths)
#extract the labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
print(labels[:4])
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network...")
model = VGG16(weights= "imagenet",include_top=False)

#initialize the hdf5 dataset riter and store data
dataset = HDF5DatasetWriter((len(imagePaths),512*7*7),args["output"],dataKey="features",bufSize=args["buffersize"])
dataset.storeClassLabels(le.classes_)

#initialize the progressbar
widgets = ["Extracting features: ",progressbar.Percentage()," ",progressbar.Bar()," ",progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths),widgets = widgets).start()

#loop over images
for i in np.arange(0,len(imagePaths),bs):
    #extract the batch of images and labels ,then initialize the list of actual images passed 
    batchPaths = imagePaths[i : bs + i]
    batchlabels = labels[i : bs + i]
    batchImages = []
    for (j,imagePath) in enumerate(batchPaths):
        image = load_img(imagePath,target_size=(224,224))
        image = img_to_array(image)

        #preprocess by expanding the dims and subtracting the mean of the rgb intensity
        image = np.expand_dims(image,axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)
    #pass the images to the network and use output as features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages,batch_size=bs)
    #reshape 
    features = features.reshape((features.shape[0],512*7*7))
    dataset.add(features,batchlabels)
    pbar.update(i)
dataset.close()
pbar.finish()