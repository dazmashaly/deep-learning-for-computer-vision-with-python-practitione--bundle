from config import d_v_c_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

#grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

#train and test split
split = train_test_split(trainPaths,trainLabels,test_size=config.NUM_TEST_IMAGES,stratify=trainLabels,random_state=42)
(trainPaths,testPaths,trainLabels,testLabels) = split
#val split
split = train_test_split(trainPaths,trainLabels,test_size=config.NUM_VAL_IMAGES,stratify=trainLabels,random_state=42)
(trainPaths,valPaths,trainLabels,valLabels) = split
#list pairing the training ,val,test image paths with thire label and output hdf5
datasets = [("train",trainPaths,trainLabels,config.TRAIN_HDF5),("val",valPaths,valLabels,config.VAL_HDF5),("test",testPaths,testLabels,config.TEST_HDF5)]
# initialize the preprocessor and the list of rgb channel averages the subtract them later
aap = AspectAwarePreprocessor(256,256)
(R,G,B) = ([],[],[])

for (dType,paths,labels,outputPath) in datasets:

    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths),256,256,3),outputPath)
    #initialize the prograss bar
    widgets = ["buliding Daraset: ",progressbar.Percentage()," ",progressbar.Bar()," ",progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()

    for(i,(path,label)) in enumerate(zip(paths,labels)):
        image = cv2.imread(path)
        image = aap.Preprocess(image)
        #if this is the traning the compute the mean
        if dType == "train":
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        #add image and label to dataset    
        writer.add([image],[label])
        pbar.update(i)
    pbar.finish()
    writer.close()
print("[INFO] serializing means...")
D = {"R":np.mean(R),"G":np.mean(G),"B":np.mean(B)}
f = open(config.DATASET_MEAN,"w")
f.write(json.dumps(D))
f.close()