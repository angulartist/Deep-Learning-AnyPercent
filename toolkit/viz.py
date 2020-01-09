import numpy as np
from matplotlib import pyplot as plt


def plot_loss(num_epochs, h):
    plt.plot(np.arange(0, num_epochs), h.history['loss'], label='train_loss')
    plt.plot(np.arange(0, num_epochs), h.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, num_epochs), h.history['acc'], label='train_accuracy')
    plt.plot(np.arange(0, num_epochs), h.history['val_acc'], label='val_accuracy')

    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.title('Training Loss/Accuracy')
    plt.legend()

    plt.show()
