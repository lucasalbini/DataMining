import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import cv2
from keras.applications.inception_v3 import preprocess_input
import os
from utils import count_images

def preprocess(image, width= 224, height = 224):
        image = cv2.resize(image,(width,height))
        image = preprocess_input(image)
        return image

def load_dataset():
        train_data_dir = 'train_images/'
        test_data_dir = 'test_images/'
        width = 224
        height = 224
        channels = 3

        n_train_images, n_test_images = count_images(train_data_dir, test_data_dir)

        train_data = np.zeros((n_train_images, width, height, channels), dtype = np.float32)
        test_data = np.zeros((n_test_images, width, height, channels), dtype = np.float32)
        train_labels = np.empty((n_train_images))
        test_labels = np.empty((n_test_images))

        i = 0
        for label, folder in enumerate(sorted(os.listdir(train_data_dir))):
                print 'currently reading train images from folder ', label,' :', folder
                for files in os.listdir(train_data_dir+folder):
                        current_image = load_img(train_data_dir+folder+'/'+files)
                        current_image = img_to_array(current_image)
                        current_image = preprocess(current_image)
                        train_data[i] = current_image
                        train_labels[i] = label
                        i+=1
        i = 0
        for label, folder in enumerate(sorted(os.listdir(test_data_dir))):
                print 'currently reading test images from folder ', label,' :', folder
                for files in os.listdir(test_data_dir+folder):
                        current_image = load_img(test_data_dir+folder+'/'+files)
                        current_image = img_to_array(current_image)
                        current_image = preprocess(current_image)
                        test_data[i] = current_image
                        test_labels[i] = label
                        i+=1
        return train_data, test_data, train_labels, test_labels


train_data, test_data, train_labels, test_labels = load_dataset()
