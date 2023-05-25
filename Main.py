import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

#initialising the CNN
classifier = Sequential()
#step1 - Convolution
from keras.layers import Dense, Dropout
classifier.add(Conv2D(128, (3,3), activation='relu',kernel_initializer='he_uniform',padding='same',input_shape=(48,48,3)))
classifier.add(MaxPooling2D(pool_size =(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_uniform',padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation= 'relu',kernel_initializer = 'he_uniform'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 7, activation = 'softmax')) #7 đầu ra cho 7 cảm xúc

classifier = load_model('model_emo.h5')

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set=train_datagen.flow_from_directory('train\\',
                                               target_size=(48,48),
                                               batch_size=64,
                                               class_mode ='categorical')
test_set=train_datagen.flow_from_directory('test\\',
                                               target_size=(48,48),
                                               batch_size=64,
                                               class_mode ='categorical')

from tensorflow.keras.callbacks import EarlyStopping

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
callbacks=[EarlyStopping(monitor='val_loss',patience=100)]

history=classifier.fit(training_set,
                  steps_per_epoch=50,
                  batch_size = 64,
                  epochs=100,
                  validation_data=test_set,
                  validation_steps=50,
                  callbacks=callbacks,
                  verbose = 1)

#đánh giá chất lượng của mô hình và vẽ lại
score = classifier.evaluate(test_set,verbose=0)
print('Sai số kiểm tra là: ',score[0])
print('Độ chính xác kiểm tra là: ',score[1])
print('Done Train')
classifier.save('model_emo.h5')
print('Done Save')