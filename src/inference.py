import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(image_path, model_path):
    model = load_model(model_path)

    # Load image
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_id, confidence

if __name__ == "__main__":
    image_path = '../data/test/example.jpg'
    model_path = '../models/mobilenet_golf_model.h5'

    class_id, confidence = predict_image(image_path, model_path)
    print(f"Predicted Class: {class_id}, Confidence: {confidence:.2f}")
