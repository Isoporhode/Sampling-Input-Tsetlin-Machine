###
# views logs
# example run: (logfile is located in logs directory)
# python3 logview.py -n dataset(0-1-RGB-GAUSSIAN-3C-fbank=No)03-03-2020T15:14:35.pkl
###
from logger import Log_open
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
import re
import pandas as pd
from statistics import mean

cifar10_labels = [
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"
]

def read_log_names(search = '*.json'):
    return glob.glob(os.path.join('logs',search))

def conf_matrix_extract(epochs):
    validation_confusion_matrix_array = []
    training_confusion_matrix_array = []
    for m in range(len(epochs)):
        validation_confusion_matrix_array.append(epochs[m]["validation_confusion_matrix"])
        training_confusion_matrix_array.append(epochs[m]["training_confusion_matrix"])
    return validation_confusion_matrix_array, training_confusion_matrix_array

def conf_matrix_split(conf_matrix):
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    return (TN, FP, FN, TP)

def accuracy_multiple_classes (conf_matrix):
    T = 0
    F = 0
    for row in range(len(conf_matrix)):
        for column in range(len(conf_matrix[0])):
            if row == column:
                T += conf_matrix[row][column]
            else:
                F += conf_matrix[row][column]
    return T/(T+F)

def accuracy(conf_matrix):
    TN, FP, FN, TP = conf_matrix_split(conf_matrix)
    return (TP + TN) / (TP + TN + FP + FN) * 1.0

def TPR(conf_matrix):
    TN, FP, FN, TP = conf_matrix_split(conf_matrix)
    return TN/(TN + FP)*1.0

def F1_score(conf_matrix):
    TN, FP, FN, TP = conf_matrix_split(conf_matrix)
    return 2*TP/(2*TP+FP+FN)

def MCC_score (conf_matrix):
    TN, FP, FN, TP = conf_matrix_split(conf_matrix)
    return ((TP * TN)-(FP * FN))/math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))*1.0

def calculate_scores(confusion_matrix_array):
    MCC_a = []
    accuracy_a = []
    f1_a = []
    for i in range(len(confusion_matrix_array)):
        MCC_a.append(MCC_score(confusion_matrix_array[i]))
        accuracy_a.append(accuracy_multiple_classes(confusion_matrix_array[i]))
        f1_a.append(F1_score(confusion_matrix_array[i]))
    return (MCC_a, accuracy_a, f1_a)

def average_last_20(score):
    return np.average(np.asarray(score)[:-20])

def average_last_100(score):
    return np.average(np.asarray(score)[:-100])

def top_20(score):
    top_array = []
    for value in score:
        if len(top_array) <= 20 or value > min(top_array):
            top_array.append(value)
            if len(top_array) > 20:
                top_array.remove(min(top_array))
    top_array.sort()
    return top_array

parser = argparse.ArgumentParser()
parser.add_argument('-n', action='store', dest='log_name')
parser.add_argument('-s', action='store', dest='search_string',
                    default='*')
parser.add_argument('--tex', action='store', dest='produce_tex', default=False)
args = parser.parse_args()

if args.log_name:
    log_opener = Log_open(args.log_name)
    log_opener.list_metadata()
    (validation_confusion_matrix_array_test, validation_confusion_matrix_array_train) = conf_matrix_extract(log_opener.get_epochs())
    (MCC_a, accuracy_a_test, f1_a) = calculate_scores(validation_confusion_matrix_array_test)
    (MCC_a, accuracy_a_train, f1_a) = calculate_scores(validation_confusion_matrix_array_train)

    # plt.plot(MCC_a, label='MCC')
    plt.plot(accuracy_a_test, label='acc_test')
    plt.plot(accuracy_a_train, label='acc_train')

    #plt.plot(f1_a, label='f1 score')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
else:
    log_files_relative_path_names = read_log_names(args.search_string)
    print("all logfiles:", log_files_relative_path_names)
    print()
    logs = []
    for relative_path_name in log_files_relative_path_names:
        log_opener = Log_open(relative_path_name)
        logs.append([relative_path_name, log_opener])
    acc_arr = []
    logs_without_trash_logs = []
    for log in logs:
        validation_confusion_matrix_array, training_confusion_matrix_array = conf_matrix_extract(np.asarray(log[1].get_epochs()))
        if len(validation_confusion_matrix_array) < 5:
            print(log[0][5:], "has less than 100 epochs!")
            continue
        # print(validation_confusion_matrix_array)
        (MCC, acc, f1) = calculate_scores(validation_confusion_matrix_array)
        acc_arr.append(average_last_100(acc))
        logs_without_trash_logs.append(log)
    acc_sorted_logs = [x for _,x in sorted(zip(acc_arr,logs_without_trash_logs))]
    pair_matrix_acc = np.zeros((10,10))
    pair_matrix_acc_top = np.zeros((10,10))
    pair_matrix_test = np.zeros((10,10))
    print("average last 100:")
    for log in acc_sorted_logs:
        validation_confusion_matrix_array, training_confusion_matrix_array = conf_matrix_extract(log[1].get_epochs())
        (MCC, acc, f1) = calculate_scores(validation_confusion_matrix_array)
        av_acc_100 = average_last_100(acc)
        av_mcc_100 = average_last_100(MCC)
        print(log[0][5:],log[1].get_metadata())
        print("Last 100 acc:", av_acc_100, "MCC:", av_mcc_100)
        print("top 20 acc:", top_20(acc))
        print()

    # Pair matrix stuff, needs 45 log-pairs to work, old code:
    # avg_pma = []
    # for x_pair in pair_matrix_acc:
    #     for y_pair in x_pair:
    #         if y_pair != 0:
    #             avg_pma.append(y_pair)
    # print("Average Acc pair matrix:", mean(avg_pma)*100)
    # print("Accuracy over the last 100 epochs for pairs:")
    # print (pd.DataFrame(pair_matrix_acc.T*100, index=cifar10_labels,
    #                     columns=cifar10_labels))
    # average_dataframe = pd.DataFrame(pair_matrix_acc.T*100, index=cifar10_labels,
    #                     columns=cifar10_labels)
    # if args.produce_tex:
    #     with open('average.tex', 'w') as tf:
    #         tf.write(average_dataframe.to_latex(float_format=lambda x: '%10.2f' % x))
    # print()
    # avg_pmat = []
    # for x_pair in pair_matrix_acc_top:
    #     for y_pair in x_pair:
    #         if y_pair != 0:
    #             avg_pmat.append(y_pair)

    # print("Average Acc top pair matrix:", mean(avg_pmat)*100)
    # print("Top epoch accuracy for each pair:")
    # print (pd.DataFrame(pair_matrix_acc_top.T*100, index=cifar10_labels,
    #                     columns=cifar10_labels))

    # top_dataframe = pd.DataFrame(pair_matrix_acc_top.T*100, index=cifar10_labels,
    #                     columns=cifar10_labels)
    # if args.produce_tex:
    #     with open('top.tex', 'w') as tf:
    #         tf.write(top_dataframe.to_latex(float_format=lambda x: '%10.2f' % x))


