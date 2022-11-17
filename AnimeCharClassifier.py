import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random
import math
import cv2

class AnimeCharClassifier:
    def __init__(self, image_dir, class_names, training_split=.8):
        self.image_dir = image_dir
        self.training_split = training_split
        self.train_dir = f"{self.image_dir}/train"
        self.test_dir = f"{self.image_dir}/test"
        self.train_data = None
        self.test_data = None
        self.model = None
        self.history = None
        self.class_names = class_names

    def _split_training_data(self):
        dir_list = os.listdir(self.image_dir)
        dir_list = list(filter(lambda x: not os.path.isdir(os.path.join(self.image_dir, x)) and not x.endswith(".json"), dir_list))
        random.shuffle(dir_list)
        last_train_index = math.floor(len(dir_list) * self.training_split) - 1
        train_files = dir_list[:last_train_index]
        test_files = dir_list[last_train_index:]

        for file in train_files:
            shutil.move(f"{self.image_dir}/{file}", f"{self.image_dir}/train/Goku/{file}")

        for file in test_files:
            shutil.move(f"{self.image_dir}/{file}", f"{self.image_dir}/test/Goku/{file}")

        self.train_dir = f"{self.image_dir}/train"
        self.test_dir = f"{self.image_dir}/test"

        return

    def _preprocess_images(self):
        train_datagen = ImageDataGenerator(rescale = 1./255)
        test_datagen = ImageDataGenerator(rescale = 1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150,150),
            batch_size=20,
            class_mode='binary'
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(150,150),
            batch_size=20,
            class_mode='binary'
        )
    
    def _generate_cnn_model(self, activation='relu', num_filters=10, kernel_size=(3, 3), strides=1, input_shape=(150,150, 3), max_pooling=(3,3)):
        model = models.Sequential()
        model.add(layers.Conv2D(num_filters, kernel_size, strides, activation=activation, input_shape=input_shape))
        model.add(layers.MaxPooling2D(max_pooling))
        model.add(layers.Dropout(0.2, input_shape=(60,)))
        model.add(layers.Conv2D(32, (3,3), 1, activation='relu'))
        model.add(layers.MaxPooling2D(max_pooling))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

        return model

    def fit(self):
        self._preprocess_images()
        self.model = self._generate_cnn_model()
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        history = self.model.fit_generator(
             self.train_generator,
             steps_per_epoch = 5,
             epochs = 50,
             validation_data = self.test_generator,
             validation_steps = 5)
        
        self.history = history
    
    def predict(self, X):
        prediction = self.model.predict(X)
        return (self.class_names[np.argmax(prediction)], prediction)

    def save_model(self):
        self.model.save(os.path.join('models', 'dbz_model.h5'))

if __name__ == "__main__":
    img_dir = "/home/mgfos207/Desktop/PetProjects/ImageScrape/AnimeCharacter"
    anime_classifier = AnimeCharClassifier(img_dir, ['Gohan', 'Goku'])
    anime_classifier.fit()
    img = cv2.imread("/home/mgfos207/Desktop/PetProjects/ImageScrape/AnimeCharacter/test-image.jpg")
    test_image = tf.image.resize(img, (150,150))
    anime_classifier.save_model()
    prediction = anime_classifier.predict(np.expand_dims(test_image/255,0))
    print(prediction)