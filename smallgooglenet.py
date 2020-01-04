from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class SmallGoogleNet(object):
    @staticmethod
    def conv_module(x, K, kX, kY, strides, chan_dim, padding='same'):
        x = Conv2D(K, (kX, kY), strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation('relu')(x)
        
        return x
    
    @staticmethod
    def inception_module(x, num_K1x1, num_K3x3, chan_dim):
        conv_1x1 = SmallGoogleNet.conv_module(x, num_K1x1, 1, 1, (1, 1), chan_dim)
        conv_3x3 = SmallGoogleNet.conv_module(x, num_K3x3, 3, 3, (1, 1), chan_dim)
            
        return concatenate([conv_1x1, conv_3x3], axis=chan_dim)
        
    @staticmethod
    def downsample_module(x, K, chan_dim):
        conv_3x3 = SmallGoogleNet.conv_module(x, K, 3, 3, (2, 2),
                                              chan_dim, padding='valid')
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
            
        return concatenate([conv_3x3, pool], axis=chan_dim)
        
    @staticmethod
    def build(width, height, num_classes, depth=3):
        input_shape = (height, width, depth)
        chan_dim = -1 # Index of the channel dimension (in order to apply BN)
        
        inputs = Input(shape=input_shape)
        x = SmallGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        x = SmallGoogleNet.inception_module(x, 32, 32, chan_dim)
        x = SmallGoogleNet.inception_module(x, 32, 48, chan_dim)
        x = SmallGoogleNet.downsample_module(x, 80, chan_dim)
        
        x = SmallGoogleNet.inception_module(x, 112, 48, chan_dim)
        x = SmallGoogleNet.inception_module(x, 96, 64, chan_dim)
        x = SmallGoogleNet.inception_module(x, 80, 80, chan_dim)
        x = SmallGoogleNet.inception_module(x, 48, 96, chan_dim)
        x = SmallGoogleNet.downsample_module(x, 96, chan_dim)
        
        x = SmallGoogleNet.inception_module(x, 176, 160, chan_dim)
        x = SmallGoogleNet.inception_module(x, 176, 160, chan_dim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)
        
        x = Flatten()(x)
        x = Dense(num_classes)(x)
        
        model = Model(inputs, x, name='googlenet')
        
        print(model.summary())
        
        return model
        