from keras.applications.vgg16 import VGG16
import argparse

print("[INFO] loading model...")
model = VGG16(weights="imagenet",include_top=True)
print("[INFO] displaying layers...")
for(i,layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i,layer.__class__.__name__))