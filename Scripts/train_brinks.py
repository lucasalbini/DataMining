import keras
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras import layers
from keras.utils import multi_gpu_model
def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004


def center_crop_generator(batches, crop_length):
  while True:
    batch_x, batch_y = next(batches)
    start_y = (img_height - crop_length) // 2
    start_x = (img_width - crop_length) // 2
    if K.image_data_format() == 'channels_last':
        batch_crops = batch_x[:, start_x:(img_width - start_x), start_y:(img_height - start_y), :]
    else:
        batch_crops = batch_x[:, :, start_x:(img_width - start_x), start_y:(img_height - start_y)]
    yield (batch_crops, batch_y)

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=[0.8,1],
        horizontal_flip=True,
        channel_shift_range = 30, 
        fill_mode='reflect',
        preprocessing_function = preprocess_input
      )

datagen_test = ImageDataGenerator(
        preprocessing_function = preprocess_input
      )

batch_size = 64

#train_batches = datagen.flow_from_directory('/home/users/datasets/notmnist/notMNIST_small/', batch_size=256, class_mode = 'categorical', target_size=(28, 28))
train_batches = datagen.flow_from_directory('train_images/', batch_size=batch_size, class_mode = 'categorical', target_size=(224, 224), shuffle=True)
test_batches = datagen_test.flow_from_directory('test_images/', batch_size=batch_size, class_mode = 'categorical', target_size=(224, 224))
#train_crops = train_batches
#train_crops = crop_generator(train_batches, 224)

'''
batch_size,batch_size,4,4,batch_s
try:
        os.makedirs('augmentation')
except:
        pass
i = 0
for batch, labels in crop_generator(train_batches, 16):
        i += 1
        #print batch.shape, batch, labels
        batch = batch[...,::-1]
        cv2.imwrite('augmentation/'+str(i)+'.jpg',np.squeeze(batch)*255)
        if i >= 200:
                break  # otherwise the generator would loop indefinitely
'''
from keras.models import Sequential
from keras import applications
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import layers

#vgg = applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224,224,3))
vgg = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,  input_shape=(224,224,3), pooling = 'avg')
model = Sequential()

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in vgg.layers:
        print layer.name
        #layer.trainable = False
vgg.summary()
model.add(vgg)
#model.add(layers.AveragePooling2D(pool_size=(5, 5)))
model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='softmax'))

'''
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))
'''

sgd = keras.optimizers.SGD(decay=1e-5, momentum=0.9, nesterov=True)
#sgd = keras.optimizers.Adam()

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'top_k_categorical_accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', 'top_k_categorical_accuracy'])

print model.summary()
lr_scheduler = LearningRateScheduler(schedule)
hist = parallel_model.fit_generator(
        generator = train_batches,
        epochs=30,
	steps_per_epoch =  20082/batch_size,
        validation_steps = 5021/batch_size,
        validation_data = test_batches,
        use_multiprocessing=False,
        workers=8,
        callbacks=[lr_scheduler]
        )

model.save_weights('weights.h5')


import json
with open('history.json', 'w') as f:
        json.dump(hist.history, f)













