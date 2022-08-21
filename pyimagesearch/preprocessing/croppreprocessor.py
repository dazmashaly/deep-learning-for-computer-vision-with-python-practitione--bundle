import numpy as np
import cv2

class CropPreprocessor:
    def __init__(self,width,height,horiz = True,inter = cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.horiz = horiz
        self.inter = inter
    
    def Preprocess(self,image):
        crops = []

        #grab the dims of the image and use them to define the corners
        #sw,sh = desierd dims and s,w are the true dims coords are start and end
        (h,w) = image.shape[:2]
        coords = [[0,0,self.width,self.height],[w - self.width,0,w,self.height],[w - self.width,h - self.height,w,h],[0,h - self.height,self.width,h]]

        #compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (w - self.height))
        coords.append([dW,dH,w - dW,h - dH])

        for (startX,startY,endX,endY) in coords:
            crop = image[startY:endY,startX:endX]
            crop = cv2.resize(crop,(self.width,self.height),interpolation=self.inter)
            crops.append(crop)

        if self.horiz :
            mirrors = [cv2.flip(c,1) for c in crops]
            crops.extend(mirrors)
        return np.array(crops)