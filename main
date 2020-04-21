import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt



covid_dataset = utils.obter_path_dataset()

imagens, labels = utils.obter_dataset(covid_dataset)
imagens = np.array(imagens) / 255.0
labels = np.array(labels)

le = LabelEncoder()
le.fit(labels)
encodedLabels = le.transform(labels)
labels = np_utils.to_categorical(encodedLabels)


batch_size   = 34
input_shape  = (150, 150, 3)
random_state = 42
alpha        = 1e-5
epoch        = 100

filepath= "../Projeto-Tcc/transferlearning_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')

callbacks = [checkpoint, lr_reduce]

(trainX, testX, trainY, testY) = train_test_split(imagens, labels, test_size=0.2, stratify=labels, random_state=random_state)

train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15)

train_datagen.fit(trainX)

data_aug = train_datagen.flow(trainX, trainY, batch_size=batch_size)


conv_base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

conv_base.summary()

set_trainable = False

for layer in conv_base.layers:
  if layer.name == 'block5_conv1':
    set_trainable = False
  if set_trainable:
    layer.trainable = False
  else:
    layer.trainable = False

conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(3, activation='softmax'))

model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(data_aug,
                    steps_per_epoch=len(trainX) // batch_size,
                    validation_data=(testX, testY),
                    validation_steps=len(testX) // batch_size,
                    callbacks=callbacks,
                    epochs=epoch)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred = model.predict(testX)
pred = np.argmax(pred,axis = 1)
y_true = np.argmax(testY,axis = 1)

cm = confusion_matrix(y_true, pred)


fig, ax = plot_confusion_matrix(conf_mat=cm,  figsize=(5, 5))
plt.show()

y_true = np.argmax(testY,axis = 1)
y_probas = model.predict(testX)
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()