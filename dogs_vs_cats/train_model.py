from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d","--db" ,required=True,help="path to hdf5")
ap.add_argument("-m","--model" ,required=True,help="path to output model")
ap.add_argument("-j","--jobs" ,type=int,default=-1,help="path to hdf5")
args = vars(ap.parse_args())
#open the hdf5 file for reading and determin the indx for train and test
db = h5py.File(args["db"],"r")
i = int(db["labels"].shape[0]*0.75)

print("[INFO] tuning hyperparameters...")
params = {"C":[0.0001,0.001,0.01,0.1,1.0]}
model = LogisticRegression(C=0.001)
model.fit(db["features"][:i],db["labels"][:i])
print("[INFO] model best paramters : {}".format(model.get_params))
print("[INFO] evaluating model")
preds = model.predict(db["features"][i:])
classNames = [x.decode("utf-8") for x in db["label_names"]]
print(classification_report(db["labels"][i:],preds,target_names=classNames))
acc = accuracy_score(db["labels"][i:],preds)
print("[INFO] accuracy = {}".format(acc))
print("[INFO] saving model...")
f = open(args["model"],"wb")
f.write(pickle.dumps(model))
f.close()
db.close()