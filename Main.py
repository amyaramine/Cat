import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
import time
import re
from sklearn.utils import shuffle
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

np.random.seed(1337)  # for reproducibility

img_rows = 224
img_cols = 224
nb_class = 2

nb_filters = 32
nb_conv = 3
nb_pool = 2

bath_size = 256
nb_epoch = 31
split = 0.3

pathTrain = ('../Dogs vs Cats/train/')
print pathTrain

def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    resized = img

    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    resized = resized.astype(np.float32, copy=False)

    return resized

def larger_model(img_rows, imgcols, color_type=1):
    # create model
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(color_type, img_rows, img_cols), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
	# Compile model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def little_Model(img_rows, imgcols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    print "model compiled"
    return model

def CNN_Model4Couches(img_rows, img_cols, color_type=1):
  model = Sequential()
  model.add(Convolution2D(16, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(color_type, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(32, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(64, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(128, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_class))
  model.add(Activation('sigmoid'))

  model.summary()
  model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
  print "model compiled"
  return model


liste = os.listdir(pathTrain)
liste = sorted(liste, key=len)
Images = []
print "train :", len(liste)

Y_train = []
i = 0
start = time.clock()

print "Load Train data"
i = 0
for element in liste:
    name = re.findall("[a-z]*", element)
    name = name[0] 
    #print name
    # if i == 3:
    #     break
    Images.append(get_im(pathTrain+element, img_rows, img_cols))
    # plt.imshow(Images[i])
    # plt.show()
    if( name =="cat"):
        Y_train.append(0)
    else:
        Y_train.append(1)
    i += 1

print "Loading train data took : ", time.clock()-start, "seconds"
X_train = np.asarray(Images)

del Images
# for element in X_train:
#     plt.imshow(element)
#     plt.show()

#train_color = np.ndarray(X_train.shape[0], img_rows, img_cols, 1), dtype=np.uint8)
X_train = X_train / 255
X_train = shuffle(X_train, random_state = 0)

#X_train = X_train[:1000]
X_train = X_train.reshape(X_train.shape[0],1, img_rows, img_cols)

Y_train = shuffle(Y_train, random_state = 0)
#Y_train = Y_train[:1000]
Y_train = np_utils.to_categorical(Y_train, nb_class)


print "Data Loaded"
print "X_train : ", X_train.shape
print "Y_train : ", Y_train.shape


print "Load Test data"
pathTest = "../Dogs vs Cats/test/"
liste = os.listdir(pathTest)
ImageTest = []

start = time.clock()
i = 0
for element in liste:
    # if i == 3:
    #     break
    ImageTest.append(get_im(pathTest+element, img_rows, img_cols))
    # plt.imshow(Images[i])
    # plt.show()
    i += 1

print "Loading test data took : ",time.clock()-start, " seconds"
X_test = np.asarray(ImageTest)
X_test = X_test / 255
del ImageTest
del liste

X_test = X_test.reshape(X_test.shape[0],1, img_rows, img_cols)
print "X_test : ", X_test.shape



model =  CNN_Model4Couches(img_rows, img_cols, 1)
model.fit(X_train, Y_train, batch_size=bath_size,
          nb_epoch=nb_epoch, verbose=1, validation_split=split, shuffle=True)

result = model.predict_classes(X_train)
print "result = ", result

score = model.evaluate(X_train, Y_train, verbose=0)
print "\nTest accuracy : ", str(score[1])

prob_result = model.predict(X_test)
#result = model.predict_classes(X_test)
# print prob_result
#print result
# prob_result = prob_result[:,0]
# print len(prob_result), prob_result
submission = open("submission.csv", 'w')
submission.write('id,label')
i = 1
#for element in result:
#    submission.write('\n'+str(i)+','+str(round(element,2)))
#    i += 1

for element in prob_result:
    submission.write('\n'+str(i)+','+str(round(element[1],2)))
    i += 1

submission.close()
