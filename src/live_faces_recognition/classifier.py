# Common imports
import os
import numpy as np

# TensorFlow imports
# may differs from version to versions

import tensorflow as tf
from tensorflow import keras

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# Test dataset is set explicitly, because the amount of data is very small
train_aug_image_folder = os.path.join('data', 'train_aug')
train_image_folder = os.path.join('data', 'train')
test_image_folder = os.path.join('data', 'test')
img_height, img_width = 128, 128  # size of images
num_classes = 2

# Training settings
validation_ratio = 0.2  # 15% for the validation

AUTOTUNE = tf.data.AUTOTUNE

class_names = ['admin', 'other']
batch_size = 16
activation = 'softmax'
optimizer = 'adam'
learning_rate = 0.5

def create_dataset():
    # Train and validation sets
    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_image_folder,
        validation_split=validation_ratio,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        label_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    val_ds = keras.preprocessing.image_dataset_from_directory(
        train_image_folder,
        validation_split=validation_ratio,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True)

    # Test set
    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_image_folder,
        image_size=(img_height, img_width),
        label_mode='categorical',
        shuffle=False)

    return (train_ds, val_ds, test_ds)

def create_aug_dataset():
    # Train and validation sets of augmented dataset
    train_aug_ds = keras.preprocessing.image_dataset_from_directory(
        train_aug_image_folder,
        validation_split=validation_ratio,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        label_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    val_aug_ds = keras.preprocessing.image_dataset_from_directory(
        train_aug_image_folder,
        validation_split=validation_ratio,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    # Test set
    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_image_folder,
        image_size=(img_height, img_width),
        label_mode='categorical',
        shuffle=False)

    return (train_aug_ds, val_aug_ds, test_ds)

# Train on augmented dataset
train_on_aug = False

if train_on_aug:
    train_ds, val_ds, test_ds = create_aug_dataset()
    name_to_save = f"models/face_classifier_aug.h5"
else:
    train_ds, val_ds, test_ds = create_dataset()
    name_to_save = f"models/face_classifier.h5"

def build_model():

    base_model = keras.applications.MobileNet(weights='imagenet',
                                                include_top=False,  # without dense part of the network
                                                input_shape=(img_height, img_width, 3))

    # Set layers to non-trainable
    for layer in base_model.layers:
        layer.trainable = False

    global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(
            num_classes, activation='sigmoid')(global_avg_pooling)

    face_classifier = keras.models.Model(inputs=base_model.input,
                                            outputs=output,
                                            name='MobileNet')
    face_classifier.summary()

    return face_classifier

def compile_model(face_classifier):

    if optimizer == 'adam':
        optimizer_object = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer_object = keras.optimizers.SGD(learning_rate=learning_rate)

    elif optimizer == 'rmsprop':
        optimizer_object = keras.optimizers.RMSprop(
            learning_rate=learning_rate)

    face_classifier.compile(loss='categorical_crossentropy',
                                optimizer=optimizer_object,
                                metrics=['accuracy'])

    return face_classifier

def train_model(face_classifier, epochs=5):

    # ModelCheckpoint to save model in case of interrupting the learning process
    checkpoint = ModelCheckpoint(name_to_save,
                                    monitor="val_loss",
                                    mode="min",
                                    save_best_only=True)

    # EarlyStopping to find best model with a large number of epochs
    earlystop = EarlyStopping(monitor='val_loss',
                                restore_best_weights=True,
                                patience=3)

    callbacks = [earlystop, checkpoint]

    history = face_classifier.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds
    )

    face_classifier.save(name_to_save)

    return history


# Dataset information
def test_image_classifier(model, path, y_true, img_height=128, img_width=128, class_names=['admin', 'other']):
    total = 0  # number of images total
    correct = 0  # number of images classified correctly

    for filename in os.listdir(path):
        if filename != ".gitignore":
            # read each image in the folder and classifies it
            test_path = os.path.join(path, filename)
            test_image = keras.utils.load_img(
                test_path, target_size=(img_height, img_width, 3))
            # from image to array, can try type(test_image)
            test_image = keras.utils.img_to_array(test_image)

            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image, verbose=0)

            y_pred = class_names[np.array(result[0]).argmax(
                axis=0)]  # predicted class

            total += 1

            if y_pred == y_true:
                correct += 1

    print("\nTotal accuracy is {:.2f}% = {}/{} samples classified correctly".format(
        correct/total*100, correct, total))


def test_model():
    model_name = 'face_classifier.h5'
    face_classifier = keras.models.load_model(f'models/{model_name}')

    test_image_classifier(face_classifier,
                                      'data/test/admin',
                                      y_true='admin')

    test_image_classifier(face_classifier,
                                      'data/test/other',
                                      y_true='other')

def train_test_model():
    face_classifier = build_model() # Build the model
    train_model(compile_model(face_classifier)) # Compile and train the model
    test_model() # Test the model on the test set
