from os import path

TRAIN_IMAGES ="D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\\data\\tiny-imagenet-200\\train"
VAL_IMAGES = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\\data\\tiny-imagenet-200\\val\\images"

VAL_MAPP = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\\data\\tiny-imagenet-200\\val\\val_annotations.txt"
WORDNET_IDS = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\data\\tiny-imagenet-200\\wnids.txt"
WORD_LABELS = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\\data\\tiny-imagenet-200\\words.txt"

NUM_CLASSES = 200
NUM_TEST_IMAGES = 50*NUM_CLASSES

TRAIN_HDF5 = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\\data\\tiny-imagenet-200\\hdf5\\train.hdf5"
VAL_HDF5 = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\\data\\tiny-imagenet-200\\hdf5\\val.hdf5"
TEST_HDF5 = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\pract_bundel\\deepgnet\\data\\tiny-imagenet-200\\hdf5\\test.hdf5"

DATA_MEAN = "D:\projeckts\deeplearning\work\DL_for_CV\pract_bundel\deepgnet\output\mean.json"
output_PATH = "D:\projeckts\deeplearning\work\DL_for_CV\pract_bundel\deepgnet\output"
MODEL_PATH = path.sep.join([output_PATH,"cheakpoints/epoch_70.hdf5"])
FIG_PATH = path.sep.join([output_PATH,"imgnet.png"])
json_path = path.sep.join([output_PATH,"img.json"])