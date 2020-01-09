from keras.callbacks import ModelCheckpoint


def best_checkpoint(path='best_weights.h5', target='val_loss'):
    return ModelCheckpoint(
        filepath=path,
        monitor=target,
        save_best_only=True,
        verbose=1
    )


def polynomial_decay(epochs, epoch, power=2.0, base_lr=5e-3):
    return base_lr * (1 - (epoch / float(epochs))) ** power
