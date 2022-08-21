#importing packages
import re
import cv2

class SimplePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        #storing the image data, inter = interpolation method used to resize
        self.width = width
        self.height = height
        self.inter = inter

    def Preprocess(self,image):
        #resize the image ignoring the aspect ratio
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)

        
