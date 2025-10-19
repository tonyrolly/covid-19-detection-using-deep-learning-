from keras.models import Sequential
from keras.layers import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D

def CNN(n_class):
    model_name = 'cnn_model'

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(150,150,3)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(32, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(64, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(128, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())    

    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(512, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(512, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(PReLU())    
    model.add(MaxPooling2D(pool_size=(2,2)))

    
    
    

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(n_class))
    model.add(Activation('softmax'))
    
    #sgd = Adam(lr=0.01, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                 )

    return model