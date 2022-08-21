import h5py
import os

from numpy import unicode_

class HDF5DatasetWriter:
    def __init__ (self,dims,outputPath,dataKey = "images" , bufSize = 1000):
        # cheak to see if output path alreay exists 
        if os.path.exists(outputPath):
            raise ValueError ("the path already exists and cant be overwritten",outputPath)
        
        # open the hdf5 database and  create two datasets:
        # one to store the images and another for the labels
        self.db = h5py.File(outputPath,"w")
        self.data = self.db.create_dataset(dataKey,dims,dtype="float")
        self.labels = self.db.create_dataset("labels",(dims[0]),dtype="int")

        # store the buff size and initialize the buffer itself
        self.bufSize = bufSize
        self.buffer = {"data":[],"labels":[]}
        self.idx = 0


    def add(self , rows,labels):
        #add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush (self):
        #write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data":[],"labels":[]}
    
    def storeClassLabels (self,classLabels):
        #create a dataset to store the actual class label names,then store them
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names",(len(classLabels),),dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) >0:
            self.flush()
        self.db.close()
