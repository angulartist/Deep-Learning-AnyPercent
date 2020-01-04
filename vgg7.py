from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense

class SmallVGGNet(object):
    @staticmethod
    def build(width, height, num_classes, depth=3): # Depth is 3 cuz' RGB color-space
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1 # Index of the channel dimension (in order to apply BN)
        
        # First layers set ((Conv => ReLU => BN) * 2 => POOL => DO)
        # 32 filters
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2))) # Implicit stride of 2x2
        model.add(Dropout(0.25))
        
        # Second layers set ((Conv => ReLU => BN) * 2 => POOL => DO)
        # 64 filters
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # FC layers set (FC => ReLU)
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Softmax Classifier
        model.add(Dense(num_classes)) # Determined by the number of classes
        model.add(Activation('softmax'))
        
        print(model.summary())
        
        return model