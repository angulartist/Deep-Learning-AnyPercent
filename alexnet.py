from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense

class AlexNet(object):
    @staticmethod
    def build(width, height, num_classes, depth=3): # Depth is 3 cuz' RGB color-space
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1 # Index of the channel dimension (in order to apply BN)
        
        # First layers set
        # 96 filters
        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3))) # Implicit stride of 2x2
        model.add(Dropout(0.25))
        
        # Second layers set
        # 256 filters
        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        
        # Third layers set
        # 384 + 384 + 256 filters
        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        
        # Fourth layers set
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Fifth layers set
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Softmax Classifier
        model.add(Dense(num_classes)) # Determined by the number of classes
        model.add(Activation('softmax'))
        
        print(model.summary())
        
        return model