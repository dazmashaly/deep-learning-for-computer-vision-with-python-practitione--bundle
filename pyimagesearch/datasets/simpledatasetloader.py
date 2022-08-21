from cProfile import label
import numpy as np 
import cv2
import os  

class SimpleDatasetLoader:
    def __init__(self,preprocessors=None):
        #store the image preprocessor
        self.preprocessors = preprocessors

        #if the preprocessor is none initialize them as an empty list
        if (self.preprocessors is None):
            self.preprocessors = []
    def load(self, imagePaths, verbose =-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input of images
        for (i,imagePath) in enumerate(imagePaths):
            # load the image and extract the label assuming the path has the formate    
            # /path/{class}/{image}.jpg
            Image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    Image = p.Preprocess(Image)

            # treat processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(Image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose ==0:
                print("[INFO] processed {}/{}".format(i+1,len(imagePaths)))
        return (np.array(data),np.array(labels))


