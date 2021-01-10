from sklearn.metrics import confusion_matrix
import numpy as np
import time
from datetime import datetime
import os
import json

class Logger ():
    def __init__(self, name_dataset, ts_version, clauses, T, s, mask =
                 'no mask', comments = "NA comments"):
        self.metadata = { "name_dataset": name_dataset, "ts_version":ts_version, "clauses":clauses, "T":T, "s":s, "mask":mask, "comments" : comments}
        self.name_dataset = name_dataset[:-4]
        self.epochs = []
        self.now = datetime.now()
    def add_epoch(self, y_true_validation, y_pred_validation, y_true_train, y_pred_train):
        self.epochs.append( {"epoch_time" : time.time(), 
            "validation_confusion_matrix" :  confusion_matrix(y_true_validation, y_pred_validation).tolist(), 
            "training_confusion_matrix" : confusion_matrix(y_true_train, y_pred_train).tolist()
            }
            )

    def make_log(self):
        self.full_log = {"metadata" : self.metadata, "epochs" : self.epochs}

    def save_log(self):
        log_folder = './logs'
        if os.path.isdir(log_folder) == False:
            os.mkdir(log_folder)
        dt_string = self.now.strftime("%d-%m-%YT%H:%M:%S")
        with open(os.path.join(log_folder, self.name_dataset+dt_string+'.json'), 'w') as f:
            json.dump({"metadata" : self.metadata, "epochs" : self.epochs}, f)

class Log_open():
    def __init__(self, log_file_name = 'logs/test.json'):
        self.filename = ''
        self.filename = log_file_name
        with open(log_file_name, 'rb') as f:
            self.full_log = json.load(f)

    def list_metadata(self):
        metadata = self.full_log["metadata"]
        print("Metadata for the log", self.filename)
        print("Name:", metadata["name_dataset"])
        print("Tsetlin type:", metadata["ts_version"])
        print("Clauses:", metadata["clauses"])
        print("T:", metadata["T"])
        print("s:", metadata["s"])
        print("Mask:", metadata["mask"])

    def get_epochs(self):
        return self.full_log.epochs
