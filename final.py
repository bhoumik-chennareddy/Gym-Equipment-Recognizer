import tensorflow as tf
from PIL import Image
import numpy as np


model = tf.keras.models.load_model('gym_equipment_recognition_model.keras')


image_path = "C:/Users/bhoum/Downloads/kettlebell.jpg"  
image = Image.open(image_path)
image = image.resize((224, 224))  
image = np.array(image) / 255.0 


predictions = model.predict(np.expand_dims(image, axis=0))


class_names = ["Exercise Balls", "Kettlebell", "Assisted Pull Up and Dip Machine", "Bench Press"]
predicted_class = np.argmax(predictions)

print("Predicted class:", class_names[predicted_class])
