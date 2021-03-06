import keras
from keras.datasets import cifar10
import numpy as np
import cv2 
import random
import pickle
import argparse
from sklearn.metrics import confusion_matrix
from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
from time import time
from logger import Logger
'''
#
#
# Image processing
#
#
'''

def extract_images_from_label(image_array, label_array, label_to_extract):
    extracted_image_array = []
    extracted_label_array = []
    for i in range(len(label_to_extract)):
        for j in range(len(image_array)):
            if(label_array[j] == label_to_extract[i]):
                extracted_image_array.append(image_array[j])
                extracted_label_array.append(int(i))

    return (extracted_image_array, extracted_label_array)

def shuffle_dataset(image_array, label_array):
    pairs = list(zip(image_array, label_array))
    random.shuffle(pairs)
    image_array_rand=[]
    label_array_rand=[]
    for i in range(len(pairs)):
       image_array_rand.append(pairs[i][0])
       label_array_rand.append(pairs[i][1])
    return (image_array_rand,label_array_rand) 

def image_array_to_grayscale(image_array):
    grayscale_image_array = []
    for i in range(len(image_array)):
        grayscale_image_array.append(cv2.cvtColor(image_array[i], cv2.COLOR_BGR2GRAY))
        if(((i/len(image_array))*100)%10 == 0):
            print("Binarizing ", len(image_array), " images", (i/len(image_array))*100, "% done")
    return np.asarray(grayscale_image_array, np.uint8)

def generate_datasets(grayscale_flag):
    print("Loading and generating the dataset")
    # Load in datasets
    (image_array_train_all,label_array_train_all), (image_array_validation_all,label_array_validation_all) = cifar10.load_data()

    # Extract only automobile and dog
    (image_array_train_extracted,label_array_train_extracted) = extract_images_from_label(image_array_train_all,label_array_train_all, [1,3])
    (image_array_validation_extracted,label_array_validation_extracted) = extract_images_from_label(image_array_validation_all,label_array_validation_all, [1,3])

    # Shuffle training dataset
    (image_array_train_shuffled,label_array_train_shuffled) = shuffle_dataset(image_array_train_extracted,label_array_train_extracted)
    if grayscale_flag:
        image_array_train_grayscale = image_array_to_grayscale(image_array_train_shuffled)
        image_array_validation_grayscale = (image_array_to_grayscale(image_array_validation_extracted))
        return (image_array_train_grayscale,label_array_train_shuffled), (image_array_validation_grayscale,label_array_validation_extracted)
    else:
        image_array_train_reshaped = np.asarray(image_array_train_shuffled)
        image_array_validation_reshaped = np.asarray(image_array_validation_extracted)
        return (image_array_train_reshaped,label_array_train_shuffled), (image_array_validation_reshaped,label_array_validation_extracted)

'''
#
#
# Tsetlin Machine training and validationing
#
#
'''

def ctm(clauses, T, s, mask,image_array_train,label_array_train,image_array_validation,label_array_validation,epochs):
    # find max and min values for the train and validation data
    image_array_train_min = np.amin(image_array_train)
    image_array_train_max = np.amax(image_array_train)
    image_array_validation_min = np.amin(image_array_validation)
    image_array_validation_max = np.amax(image_array_validation)
    print(image_array_train_min,image_array_train_max,image_array_validation_min,image_array_validation_max)
    tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (mask, mask))
    log_creator = Logger("CIFAR_random_samling_validation",  "CTM", clauses, T, s, (mask, mask))
    max_acc = 0

    print(f"Running CTM with random sampling for {epochs} epochs, with the following parameters:")
    print(f"Clauses: {clauses}, s: {s}, T: {T}, mask: {mask} x {mask}")

    for i in range(epochs):
        
        start = time()
        # Generate matrix with size of the image array, with random values ranging from 0 to 255 
        random_train_matrix = np.random.randint(image_array_train_min,image_array_train_max, size=(image_array_train.shape))
        random_validation_matrix = np.random.randint(image_array_validation_min,image_array_validation_max, size=(image_array_validation.shape))

        # Sampeled images
        floaty_train_images = np.greater(random_train_matrix,image_array_train)
        floaty_validation_images = np.greater(random_validation_matrix,image_array_validation)
        
        # Fit
        tm.fit(floaty_train_images,label_array_train, epochs=1, incremental=True)
        stop = time()

        # Predict
        pred_validation = tm.predict(floaty_validation_images)
        pred_train = tm.predict(floaty_train_images)

        # Get some nice logging in terminal
        accuracy_validation = 100*(pred_validation == np.asarray(label_array_validation)).mean()
        if max_acc < accuracy_validation:
            max_acc = accuracy_validation
        conf_matrix_validation = confusion_matrix(np.asarray(label_array_validation), pred_validation)
        print('sum predict:', sum(pred_validation), 'sum validation:', sum(label_array_validation))
        print("#%d Accuracy on training data: %.2f%% (%.2fs), best accuracy on training data: %.2f%% " % (i+1, accuracy_validation, stop-start, max_acc))
        print('Confusion matrix:')
        print(conf_matrix_validation)

        # Save to logfile
        log_creator.add_epoch(np.asarray(label_array_validation), pred_validation, np.asarray(label_array_train), pred_train)
        log_creator.save_log()
    print('done')

parser = argparse.ArgumentParser()
parser.add_argument('--clauses', type=int, default=4000, help="Amount of clauses (int)")
parser.add_argument('-T', type=int, default=75, help="T variable (int)")
parser.add_argument('-s', type=float, default=10.0, help="S variable (float)")
parser.add_argument('--mask', type=int, default=10, help="int defining mask size")
parser.add_argument('--grayscale', type=bool, default=False, help="Images set to RGB or Grayscale (Boolean)")
parser.add_argument('--epochs', type=int, default=400, help="Amount of epochs to run (int)")
args = parser.parse_args()



(image_array_train,label_array_train),(image_array_validation,label_array_validation) = generate_datasets(args.grayscale)

ctm(args.clauses, args.T, args.s, args.mask,image_array_train,label_array_train,image_array_validation,label_array_validation, args.epochs)

