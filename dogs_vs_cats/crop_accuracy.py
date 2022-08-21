
from matplotlib.pyplot import axis
from config import d_v_c_config as config
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.meanPreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor
from pyimagesearch.io.hdf5datagenerateor import Hdf5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import progressbar
import json

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227,227)
mp = MeanPreprocessor(means["R"],means["G"],means["B"])
cp = CropPreprocessor(227,227)

iap = ImageToArrayPreprocessor()

print("[INFO] loading data...")
model = load_model(config.MODEL_PATH)

print("[INFO] predicting using og data")
testGen = Hdf5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[sp,mp,iap],classes=2)
preds = model.predict_generator(testGen.generator(),steps = testGen.numImages//64,max_queue_size = 64*2)

(rank1,_) = rank5_accuracy(preds,testGen.db["labels"])
print("rank 1 acc : {:.2f}%".format(rank1*100))
testGen.close()
testGen = Hdf5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[mp],classes=2)
preds = []
widgets = ["evaluating: ",progressbar.Percentage()," ",progressbar.Bar()," ",progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval= testGen.numImages //64,widgets=widgets).start()

for(i ,(images,labels)) in enumerate(testGen.generator(passes=1)):
    for image in images:
        crops = cp.Preprocess(image)
        crops = np.array([img_to_array(c) for c in crops],dtype="float32")

        pred = model.predict(crops)
        preds.append(pred.mean(axis=0))
    pbar.update(i)

pbar.finish()
print("[INFO] preds with crops...")
(rank1,_) = rank5_accuracy(preds,testGen.db["labels"])
print("rank 1 acc : {:.2f}%".format(rank1*100))

