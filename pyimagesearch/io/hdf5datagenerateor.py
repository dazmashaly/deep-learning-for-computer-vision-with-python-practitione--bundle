from ast import Not
from keras.utils import np_utils
import numpy as np
import h5py

class Hdf5DatasetGenerator:
    def __init__(self,dbPath,batchSize,preprocessors =None,aug=None,binarize = True,classes = 2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.calsses = classes
        #open the database and determin the number of entries
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self,passes=np.inf):
        epochs = 0

        #loop until we reach the desired number of epochs
        while epochs < passes:
            #loop over the dataset
            for i in np.arange(0,self.numImages,self.batchSize):
                images = self.db["images"][i: i+self.batchSize]
                labels = self.db["labels"][i: i+self.batchSize]
                if self.binarize:
                    labels = np_utils.to_categorical(labels,self.calsses)
                
                if self.preprocessors is not None:
                    procimage =[]
                    for image in images:
                        for p in self.preprocessors:
                            image = p.Preprocess(image)
                        procimage.append(image)
                    images = np.array(procimage)
                
                if self.aug is not None:
                    (images,labels) = next(self.aug.flow(images,labels,batch_size = self.batchSize))
                
                yield (images,labels)
            epochs+=1
        
    def close(self):
        self.db.close()
        

    
