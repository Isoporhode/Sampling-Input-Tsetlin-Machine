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

global cifar_labels
cifar_labels= {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


parser = argparse.ArgumentParser()
parser.add_argument('--clauses', type=int, default=4000)
parser.add_argument('-T', type=int, default=75)
parser.add_argument('-s', type=float, default=10.0)
parser.add_argument('--mask', type=int, default=10)
parser.add_argument('--grayscale', type=bool, default=False)
args = parser.parse_args()

def extract_images_from_label(image_array, label_array, label_to_extract):
    extracted_image_array = []
    extracted_label_array = []
    for i in range(len(label_to_extract)):
        for j in range(len(image_array)):
            if(label_array[j] == label_to_extract[i]):
                extracted_image_array.append(image_array[j])
                extracted_label_array.append(int(i))

    return (extracted_image_array, extracted_label_array)

def shuffle_dataset(x_dataset, y_dataset):
    pairs = list(zip(x_dataset, y_dataset))
    random.shuffle(pairs)
    x_rand=[]
    y_rand=[]
    for i in range(len(pairs)):
        x_rand.append(pairs[i][0])
        y_rand.append(pairs[i][1])
    return (x_rand, y_rand) 

def rgbTranspose(x_array):
    return np.asarray(x_array).reshape((len(x_array), 3, 32, 32)).transpose(0,2,3,1)

def image_array_to_grayscale(array_of_images):

    array_of_grayscale_images = []
    for i in range(len(array_of_images)):
        array_of_grayscale_images.append(cv2.cvtColor(array_of_images[i], cv2.COLOR_BGR2GRAY))
        if(((i/len(array_of_images))*100)%10 == 0):
            print("Binarizing ", len(array_of_images), " images", (i/len(array_of_images))*100, "% done")
    return np.asarray(array_of_grayscale_images, np.uint8)

def generate_datasets(grayscale_flag):
    # Load in datasets
    (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()

    # Extract only automobile and dog
    (x_train_extracted, y_train_extracted) = extract_images_from_label(x_train_all, y_train_all, [1,3])
    (x_test_extracted, y_test_extracted) = extract_images_from_label(x_test_all, y_test_all, [1,3])

    # Shuffle training dataset
    (x_train_shuffled, y_train_shuffled) = shuffle_dataset(x_train_extracted, y_train_extracted)
    if grayscale_flag:
        x_train_grayscale = image_array_to_grayscale(x_train_shuffled)
        x_test_grayscale = (image_array_to_grayscale(x_test_extracted))
        return (x_train_grayscale, y_train_shuffled), (x_test_grayscale, y_test_extracted)
    else:
        x_train_reshaped = rgbTranspose(x_train_shuffled)
        x_test_reshaped = rgbTranspose(x_test_extracted)
        return (x_train_reshaped, y_train_shuffled), (x_test_reshaped, y_test_extracted)

'''
#
#
# Tsetlin Machine training and testing
#
#
'''

def ctm(clauses, T, s, mask, x_train, y_train, x_test, y_test):
    # find max and min values for the train and validation data
    x_train_min = np.amin(x_train)
    x_train_max = np.amax(x_train)
    x_test_min = np.amin(x_test)
    x_test_max = np.amax(x_test)
    print(x_train_min, x_train_max, x_test_min, x_test_max)
    tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (mask, mask))
    log = Logger("CIFAR_random_samling_test", x_train, "CTM", clauses, T, s, (mask, mask))

    print('predicting 400 epochs')
    for i in range(400):
        
        start = time()
        # Generate matrix with size of the image array, with random values ranging from 0 to 255 
        random_train_matrix = np.random.randint(x_train_min, x_train_max, size=(x_train.shape))
        random_test_matrix = np.random.randint(x_test_min, x_test_max, size=(x_test.shape))
        # Returns 
        floaty_train_images = np.greater(random_train_matrix, x_train)
        floaty_test_images = np.greater(random_test_matrix, x_test)
        
        tm.fit(floaty_train_images, y_train, epochs=1, incremental=True)
        stop = time()
        
        pred = tm.predict(floaty_test_images)
        conf_matrix = confusion_matrix(np.asarray(y_test), pred)
        print('sum predict:', sum(pred), 'sum validation:', sum(y_test))
        accuracy = 100*(pred == np.asarray(y_test)).mean()
        print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, accuracy, stop-start))
        print('Confusion matrix:')
        print(conf_matrix)
        log.add_epoch(np.asarray(y_test), pred)
        log.save_log()
    print('done')

(x_train, y_train),(x_test, y_test) = generate_datasets(args.grayscale)

ctm(args.clauses, args.T, args.s, args.mask, x_train, y_train, x_test, y_test)

