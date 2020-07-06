import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *


#Image preparation

train_path = '/content/train'
test_path = '/content/test'
valid_path = '/content/valid'

test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes = ['dog', 'cat'], batch_size=10)
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes = ['dog', 'cat'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes = ['dog', 'cat'], batch_size=4)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
  if type(ims[0]) is np.ndarray:
    ims = np.array(ims).astype(np.uint8)
    if(ims.shape[-1] !=3):
      ims = ims.transpose((0,2,3,1))
  f = plt.figure(figsize=figsize)
  cols = len(ims)//rows if len(ims)%2 == 0 else len(ims)//rows+1
  for i in range(len(ims)):
    sp = f.add_subplot(rows, cols, i+1)
    sp.axis('Off')
    if titles is not None:
      sp.set_title(titles[i], fontsize=16)
    plt.imshow(ims[i], interpolation=None if interp else 'none')

imgs, labels = next(train_batches)
plots(imgs, titles =labels)

#CNN 
#First model

model = Sequential([
                    Conv2D(32, (3,3), activation='relu', input_shape = (224,224,3)),
                    Flatten(),
                    Dense(2, activation='softmax')                 
])

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=4,
                    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)


#Prediction

test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:, 0]
#print(test_labels)

predictions = model.predict_generator(test_batches, steps=1, verbose=0)


#VGG16 model

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

#importing VGG16 layers to Sequential model
model1 = Sequential()
for layer in vgg16_model.layers:
  model1.add(layer)

#model1.summary()
#removing last layer from model1
model1.layers.pop()

#model1.summary()

for layer in model1.layers:
  layer.trainable = False

#adding a new layer which classifies into 2 categories
model1.add(Dense(2, activation='softmax'))
#model1.summary()

#VGG16 Train
model1.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=4,
                    validation_data=valid_batches, validation_steps=4,
                    epochs=5, verbose=2)


#VGG16 Predict
predictions1 = model1.predict_generator(test_batches, steps=1, verbose=0)
print(predictions1)


#Confusion matrix
cm = confusion_matrix(test_labels, np.round(predictions1[:,0]))
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest',cmap=cmap )
  plt.title(title)
  plt.colorbar()
  tick_marks=np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
  else:
    print('Confusion matrix without normalization')
  
  print(cm)

  thresh = cm.max()/2
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i,j],
             horizontalalignment='center',
             color='white' if cm[i,j] > thresh else 'black')
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted value')
