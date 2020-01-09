from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.image import extract_patches_2d


def normalize01(arr):
    return arr.astype('float32') / 255.0


def get_one_patch(image, dims):
    return extract_patches_2d(
        image,
        patch_size=dims,
        max_patches=1
    )[0]


def generator(h_flip=True, v_flip=False):
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        fill_mode='nearest'
    )
