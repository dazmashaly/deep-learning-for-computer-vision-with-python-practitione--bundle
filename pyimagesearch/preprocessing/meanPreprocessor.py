import cv2

class MeanPreprocessor:
    def __init__(self,rMean,gMean,bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def Preprocess(self, image):
        #split image into RGB form
        (B,G,R) = cv2.split(image.astype("float32"))
        
        #subtract the mean for each channel
        B -= self.bMean
        G -= self.gMean
        R -= self.rMean

        #merge the channel together
        return cv2.merge([B,G,R])