import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_dir = 'C:/Users/bhoum/OneDrive/Desktop/Machine Learning/1 Gym Equipment.v1i.retinanet/'

train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')


train_labels = pd.read_csv(os.path.join(train_dir, '_annotations.csv'), header=None)
valid_labels = pd.read_csv(os.path.join(valid_dir, '_annotations.csv'), header=None)


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=train_dir,
        x_col=0,  
        y_col=5,  
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_labels,
        directory=valid_dir,
        x_col=0,  
        y_col=5,  
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')



model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),  
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax') 
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)


model.save('gym_equipment_recognition_model.keras')

print("Model saved successfully!")


#NEW CODEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a test data generator for the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale pixel values

# Create a test generator using a test dataset
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_labels,  # DataFrame containing test set labels
    directory=test_dir,  # Directory where test set images are stored
    x_col=0,  # Column with image file names
    y_col=5,  # Column with class labels
    target_size=(224, 224),  # Resized image dimensions
    batch_size=32,  # Number of images per batch
    class_mode='categorical'  # Multi-class categorical data
)

# Load the trained model if not already loaded
model = tf.keras.models.load_model('gym_equipment_recognition_model.keras')

# Evaluate the model on the testing set
test_loss, test_accuracy = model.evaluate(test_generator)

# Print the test accuracy and loss
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
