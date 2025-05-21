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


if predicted_class == "Kettlebell":
    print("Suggested Exercises for Kettlebell:")

    print("1. Kettlebell Swing")
    print("   - Stand with feet shoulder-width apart, holding the kettlebell with both hands.")
    print("   - Bend your knees slightly and hinge at your hips.")
    print("   - Swing the kettlebell back between your legs, then thrust your hips forward to swing it up to shoulder height.")
    print("   - Repeat this swinging motion in a controlled manner.")


    print("2. Goblet Squat")
    print("   - Hold the kettlebell by its handle close to your chest, with elbows pointing down.")
    print("   - Keep your feet shoulder-width apart and toes slightly outward.")
    print("   - Lower into a squat by bending your knees and hips, keeping your chest upright.")
    print("   - Push through your heels to return to the standing position.")

elif predicted_class == "Exercise Balls":
    print("Suggested Exercises for Exercise Balls:")

    print("1. Plank on Exercise Ball")
    print("   - Place your forearms on the exercise ball and extend your legs behind you.")
    print("   - Keep your body in a straight line from head to heels.")
    print("   - Hold this position for as long as you can while keeping your core engaged.")


    print("2. Ball Pass")
    print("   - Lie on your back, holding the exercise ball with your feet.")
    print("   - Lift your legs to pass the ball to your hands, then lower your legs and arms.")
    print("   - Continue passing the ball between your hands and feet in a controlled manner.")


elif predicted_class == "Assisted Pull Up and Dip Machine":
    print("Suggested Exercises for Assisted Pull Up and Dip Machine:")

    print("1. Assisted Pull Up")
    print("   - Adjust the machine's counterweight to your desired level of assistance.")
    print("   - Grasp the pull-up bar with both hands, palms facing forward.")
    print("   - Pull yourself up until your chin is above the bar, then lower back down.")
    

    print("2. Assisted Dip")
    print("   - Grasp the dip bars with both hands and press yourself up, bending your knees to rest on the pad.")
    print("   - Lower your body by bending your elbows, then push back up to the starting position.")

elif predicted_class == "Bench Press":
    print("Suggested Exercises for Bench Press:")

    print("1. Bench Press")
    print("   - Lie on the bench with feet flat on the floor.")
    print("   - Grip the barbell with hands slightly wider than shoulder-width.")
    print("   - Lower the barbell to your chest, then push it back up.")
    

    print("2. Incline Bench Press")
    print("   - Adjust the bench to an incline position.")
    print("   - Grip the barbell with hands slightly wider than shoulder-width.")
    print("   - Lower the barbell toward your upper chest, then push it back up.")