import numpy as np
import keras
from keras.applications.inception_v3 import preprocess_input
from keras import applications
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def plot_confusion_matrix(cm, classes='',
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi = 300)




def get_mapping(x, class_dict):
        return class_dict.keys()[class_dict.values().index(x)]

vgg = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,  input_shape=(224,224,3), pooling = 'avg')

model = Sequential()
model.add(vgg)
model.add(layers.Dropout(0.5))
model.add(layers.Dense(101, activation='softmax'))

model.load_weights('weights.h5')

datagen_test = ImageDataGenerator(
        preprocessing_function = preprocess_input
      )



test_batches = datagen_test.flow_from_directory('test_images/', batch_size=512, class_mode = 'categorical', target_size=(224, 224), shuffle=False)


print test_batches.classes # aqui estao os labels?
print test_batches.class_indices # dict mapping

inv_map = {v: k for k, v in test_batches.class_indices.iteritems()}
print inv_map

save_obj(test_batches.class_indices, 'class_dict')
save_obj(inv_map, 'inverse_class_dict')


sadsad = load_obj('class_dict')
print sadsad
'''
predictions = model.predict_generator(test_batches, workers=8, use_multiprocessing=True, max_queue_size = 5120)
print predictions, predictions.shape

np.save('predictions.npy', predictions)

'''

predictions = np.load('predictions.npy')

print 'accuracy: ', accuracy_score(test_batches.classes, np.argmax(predictions,axis=1))


for i in range(predictions.shape[0]):
        print 'top 5 predictions for image ', i,' : (', test_batches.class_indices.keys()[test_batches.class_indices.values().index(test_batches.classes[i])],')\n'
        current_image = predictions[i]
        sorted_indices = current_image.argsort()[-5:][::-1]
        for pred in sorted_indices:
                print test_batches.class_indices.keys()[test_batches.class_indices.values().index(pred)], ' -- probability : ', current_image[pred]
        print '\n'

#plot_confusion_matrix(confusion_matrix(test_batches.classes, np.argmax(predictions,axis=1)))
