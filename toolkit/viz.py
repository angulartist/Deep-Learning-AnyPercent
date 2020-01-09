import numpy as np
from matplotlib import pyplot as plt


def plot_images_grid(images, rows=5, cols=4):
    fig_size = [6, 8]

    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=fig_size
    )

    for index, axi in enumerate(ax.flat):
        axi.imshow(images[index])
        axi.set_title(f'Image #{index}')

    plt.tight_layout(True)
    plt.show()


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
