from cProfile import label
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath = None, startAt = 0):
        #store the output path for the figure, path to the json serilized file
        super(TrainingMonitor , self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
    
    def on_train_begin(self,logs ={}):
        # initialize the history dictionary
        self.H = {}
        # if the json history path exists , load the trainig history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt >0:
                    #loop over the entries in the history log and trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]
    

    def on_epoch_end(self, epoch, logs={}):
        for (k,v) in logs.items():
            l = self.H.get(k,[])
            l.append(v)
            self.H[k] =l
        # chek to see if history should be serialized
        if self.jsonPath is not None:
            f = open(self.jsonPath,"w")
            f.write(json.dumps(self.H))
            f.close
        # ensure that atleast 2 epochs passed before plotting
        if len(self.H["loss"]) >1:
            N = np.arange(0,len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N,self.H["loss"], label="training loss")
            plt.plot(N,self.H["val_loss"],label="validation loss")
            plt.plot(N,self.H["accuracy"],label="training acc")
            plt.plot(N,self.H["val_accuracy"],label="validation acc")
            plt.title("training loss and accuracy [epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("epoch")
            plt.legend()
            plt.savefig(self.figPath)
            plt.close()