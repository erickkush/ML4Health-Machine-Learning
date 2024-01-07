import tensorflow as tf
import numpy as np

# Function to load and preprocess an image for prediction
def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array /= 255.0  # Normalize the pixel values
    return img_array

# Function to make a prediction using the loaded model
def predict_disease_loaded_model(image_path, loaded_model, class_names):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Make the prediction using the loaded model
    prediction = loaded_model.predict(img_array)
    disease_class = np.argmax(prediction)

    # Map class index to disease name
    predicted_disease = class_names[disease_class]

    return predicted_disease

if _name_ == "_main_":
    # Load the saved model for predictions
    loaded_model = tf.keras.models.load_model('C:\\Users\\25470\\PycharmProjects\\myproject\\model\\trainedmodel.keras')

    # Define class names
    class_names = ['bronchiectasis', 'lung', 'pneumonia']

    # Example usage with the loaded model
    image_path = (r'C:\Users\25470\Desktop\lung test\person1_virus_7.jpeg'
                  r'')  # Replace with the path to your image
    prediction = predict_disease_loaded_model(image_path, loaded_model, class_names)

    print(f"Prediction for {image_path}: {prediction}")
