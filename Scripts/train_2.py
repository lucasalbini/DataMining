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
        return .1
    elif epoch < 28:
        return .02
    else:
        return .004

#ler dados
import load_data

x_train, x_test, y_train, y_test = load_data.load_dataset()

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
        #preprocessing_function = preprocess_input
      )

datagen_test = ImageDataGenerator(
        #preprocessing_function = preprocess_input
      )

datagen.fit(x_train)
datagen_test.fit(x_test)

batch_size = 64

#train_batches = datagen.flow_from_directory('train_images/', batch_size=batch_size, class_mode = 'categorical', target_size=(224, 224), shuffle=True)
#test_batches = datagen_test.flow_from_directory('test_images/', batch_size=batch_size, class_mode = 'categorical', target_size=(224, 224))


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

model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='softmax'))


sgd = keras.optimizers.SGD(decay=1e-5, momentum=0.9, nesterov=True)
#sgd = keras.optimizers.Adam()

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'top_k_categorical_accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', 'top_k_categorical_accuracy'])

print model.summary()
lr_scheduler = LearningRateScheduler(schedule)

hist = parallel_model.fit_generator(
        generator = datagen.flow(x_train, y_train, batch_size=32),
        epochs=30,
	steps_per_epoch =  x_train.shape[0] /batch_size,
        validation_steps = x_test.shape[0] / batch_size,
        validation_data = datagen_test.flow(x_test, y_test, batch_size=32),
        use_multiprocessing=False,
        workers=8,
        callbacks=[lr_scheduler]
        )

model.save_weights('weights.h5')


import json
with open('history.json', 'w') as f:
        json.dump(hist.history, f)













