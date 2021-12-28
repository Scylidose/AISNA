# Common imports
import os
import csv

# TensorFlow imports
# may differs from version to versions

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


train_aug_image_folder = os.path.join('data', 'train_aug')
train_image_folder = os.path.join('data', 'train')
test_image_folder = os.path.join('data', 'test')
img_height, img_width = 128, 128  # size of images
num_classes = 2 

validation_ratio = 0.2

AUTOTUNE = tf.data.AUTOTUNE

class_names = ['admin', 'other']
batch_size_array = [16, 64, 128, 256]
activation_array = ['sigmoid', 'relu', 'softmax']
optimizer_array = ['adam', 'rmsprop', 'sgd']
learning_rate_array = [0.001, 0.01, 0.05, 0.1, 0.5]

for batch_size in batch_size_array:
    for activation in activation_array:
        for optimizer in optimizer_array:
            for learning_rate in learning_rate_array:
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

                # Train and validation sets of augmented dataset
                train_aug_ds = keras.preprocessing.image_dataset_from_directory(
                    train_aug_image_folder,
                    validation_split=validation_ratio,
                    subset="training",
                    seed=42,
                    image_size=(img_height, img_width),
                    label_mode='categorical',
                    batch_size=batch_size,
                    shuffle=True)

                val_aug_ds = keras.preprocessing.image_dataset_from_directory(
                    train_aug_image_folder,
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

                base_model = keras.applications.MobileNet(weights='imagenet',
                                                        include_top=False,  # without dense part of the network
                                                        input_shape=(img_height, img_width, 3))

                # Set layers to non-trainable
                for layer in base_model.layers:
                    layer.trainable = False


                global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
                output = keras.layers.Dense(num_classes, activation='sigmoid')(global_avg_pooling)

                face_classifier = keras.models.Model(inputs=base_model.input,
                                            outputs=output,
                                            name='MobileNet')
                # face_classifier.summary()

                train_on_aug = False  # train on augmented dataset

                if train_on_aug:
                    train_ds = train_aug_ds
                    val_ds = val_aug_ds

                if train_on_aug:
                    name_to_save = f"models/test_face_classifier_aug.h5"
                else:
                    name_to_save = f"models/test_face_classifier.h5"

                # ModelCheckpoint to save model in case of interrupting the learning process
                checkpoint = ModelCheckpoint(name_to_save,
                                            monitor="val_loss",
                                            mode="min",
                                            save_best_only=True,
                                            verbose=0)

                # EarlyStopping to find best model with a large number of epochs
                earlystop = EarlyStopping(monitor='val_loss',
                                        restore_best_weights=True,
                                        patience=3,  # number of epochs with no improvement after which training will be stopped
                                        verbose=0)

                callbacks = [earlystop, checkpoint]

                if optimizer == 'adam':
                    optimizer_object = keras.optimizers.Adam(learning_rate=learning_rate)
                elif optimizer == 'sgd':
                    optimizer_object = keras.optimizers.SGD(learning_rate=learning_rate)

                elif optimizer == 'rmsprop':
                    optimizer_object = keras.optimizers.RMSprop(learning_rate=learning_rate)

                face_classifier.compile(loss='categorical_crossentropy',
                                        optimizer=optimizer_object,
                                        metrics=['accuracy'])

                epochs = 5
                history = face_classifier.fit(
                    train_ds,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=val_ds
                )
                print("BATCH SIZE:", batch_size)
                print("ACTIVATION:", activation)
                print("OPTIMIZER:", optimizer)
                print("LEARNING RATE:", learning_rate)
                print("\n--------------\n")
                print("loss: ", history.history['loss'])
                print("accuracy: ", history.history['accuracy'])
                print("val_loss: ", history.history['val_loss'])
                print("val_accuracy: ", history.history['val_accuracy'])
                print("\n--------------\n")

                with open('data/fine_tuning.csv', 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['MobileNet', batch_size, activation, optimizer, learning_rate, history.history['loss'][0], history.history['accuracy'][0], history.history['val_loss'][0], history.history['val_accuracy'][0]])

