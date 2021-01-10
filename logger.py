from sklearn.metrics import confusion_matrix
import numpy as np
import time
import pickle
from datetime import datetime
import os
import json
# for whole log: [[metadata], [epoch_nr, time_finished, [confusion_matrix]]]
# metadata: [[name_x_train, name_y_train, name_x_val, name_y_val], [shape of all
# lists]]
# metadata is written once in the start, each epoch gets their own list with
# add_epoch


class Logger ():
    def __init__(self, name_dataset, ts_version, clauses, T, s, mask =
                 'no mask', comments = "NA comments"):
        self.metadata = { "name_dataset": name_dataset, "ts_version":ts_version, "clauses":clauses, "T":T, "s":s, "mask":mask, "comments" : comments}
        self.name_dataset = name_dataset[:-4]
        self.frames = []
        self.log = {}
        self.log_open = []
        self.now = datetime.now()
    def add_epoch(self, y_true_test, y_pred_test, y_true_train, y_pred_train):
        self.frames.append( {"epoch_time" : time.time(), 
            "validation_confusion_matrix" :  confusion_matrix(y_true_test, y_pred_test).tolist(), "training_confusion_matrix" : confusion_matrix(y_true_train, y_pred_train).tolist()})

    def make_log(self):
        self.log = {"metadata" : self.metadata, "epochs" : self.frames}

    def save_log(self):
        log_folder = './logs'
        if os.path.isdir(log_folder) == False:
            os.mkdir(log_folder)
        dt_string = self.now.strftime("%d-%m-%YT%H:%M:%S")
        self.make_log()
        print(self.log)
        with open(os.path.join(log_folder, self.name_dataset+dt_string+'.json'), 'w') as f:
            json.dump(self.log, f)

class Log_open():
    def __init__(self):
        self.log = []
        self.filename = ''
    def open_log(self, log_file='test.pkl'):
        self.filename = log_file
        with open(log_file, 'rb') as f:
            self.log = pickle.load(f)
    def list_metadata(self):
        #print(self.log)
        print("Metadata for the log", self.filename)
        print("Name:", self.log[0][0][0])
        print("Tsetlin type:", self.log[0][0][1])
        print("Clauses:", self.log[0][0][2])
        print("T:", self.log[0][0][3])
        print("s:", self.log[0][0][4])
        print("Mask:", self.log[0][0][5])

dataset = [[3,4],[5,6]]
log = Logger("test_navn4_.pkl",  "test_tsetlin", 2000, 50, 10.0, (2,3))
log.add_epoch([1,0,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1])
log.add_epoch([1,0,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1])
log.add_epoch([1,0,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1])
log.add_epoch([1,0,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1])

log.save_log()
#log_opener = Log_open()
