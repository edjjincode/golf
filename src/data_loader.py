import os
import json
import tensorflow as tf

def load_data(data_dir):
    annotations_file = os.path.join(data_dir, "_annotations.coco.json")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    images = []
    labels = []

    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        image_info = next(img for img in annotations['images'] if img['id'] == image_id)
        image_path = os.path.join(data_dir, image_info['file_name'])

        # Load and preprocess the image
        image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        image = tf.keras.utils.img_to_array(image) / 255.0

        images.append(image)
        labels.append(category_id)

    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)
