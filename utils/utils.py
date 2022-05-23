# Common imports
import os
import numpy as np
# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports
# may differs from version to versions

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_augmentation():
    # Dataset information
    image_folder = os.path.join('data', 'train')
    img_height, img_width = 128, 128  # size of images

    dataset = keras.preprocessing.image_dataset_from_directory(
                image_folder,
                seed=42,
                image_size=(img_height, img_width),
                label_mode='categorical',
                shuffle=True
            )
    
    n = 5

    aug_image_folder = os.path.join('data', 'train_aug')
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.7, 1),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')


    class_names = ["admin"] # dataset.class_names

    for class_name in class_names:
        # classes: 'me' and 'not_me'
        image_folder_to_generate = os.path.join(image_folder, class_name)
        image_folder_to_save = os.path.join(aug_image_folder, class_name)
        if not os.path.exists(image_folder_to_save):
            os.makedirs(image_folder_to_save)  # create folder if doesn't exist

        i = 0
        total = len(os.listdir(image_folder_to_generate))  # number of files in folder
        for filename in os.listdir(image_folder_to_generate):
            # for each image in folder: read it
            if filename != ".gitignore":
                image_path = os.path.join(image_folder_to_generate, filename)

                image = keras.preprocessing.image.load_img(
                    image_path, target_size=(img_height, img_width, 3))
                image = keras.preprocessing.image.img_to_array(
                    image)  # from image to array

                image = np.expand_dims(image, axis=0)

                # create ImageDataGenerator object for it
                current_image_gen = train_datagen.flow(image,
                                                    batch_size=1,
                                                    save_to_dir=image_folder_to_save,
                                                    save_prefix=filename,
                                                    save_format="jpg")

                # generate n samples
                count = 0
                for image in current_image_gen:  # accessing the object saves the image to disk
                    count += 1
                    if count == n:  # n images were generated
                        break
                i += 1

        print("\nTotal number images generated for "+class_name+"= {}".format(n*total))
