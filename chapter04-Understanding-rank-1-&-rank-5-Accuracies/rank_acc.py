from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument("-d","--db",required=True)
ap.add_argument("-m","--model",required=True)
args = vars(ap.parse_args())

db = h5py.File(args["db"],"r")
i = int(db["labels"].shape[0]*0.75)

model = pickle.loads(open(args["model"],"rb").read())


print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
classNames = [x.decode("utf-8") for x in db["label_names"]]
(rank1,rank5) = rank5_accuracy(preds,db["labels"][i:])

print("[INFO] rank1: {:.2f}%".format(rank1*100))
print("[INFO] rank5: {:.2f}%".format(rank5*100))
db.close()