from keras.utils import to_categorical


def one_hot_encode(label_types, num_classes):
    return (to_categorical(labels, num_classes=num_classes) for labels in label_types)
