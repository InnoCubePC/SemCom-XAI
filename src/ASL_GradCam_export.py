# Multiple images
# Input folder: images
# Output folder: gradcam_outputs

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import preprocess

### Placeholder ###
model = tf.keras.models.load_model("models/trained_model.model") # Load the pre-trained model
### Placeholder ###

# Define the class labels
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]


# Function to predict the class of the input image
def predict_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found or could not be read.")
    
    # Preprocess the image
    preprocessed_image = preprocess.preprocess_image(image)
    # Make predictions
    predictions = model.predict(preprocessed_image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, predicted_class_index, preprocessed_image

# GradCAM Implementation for multiple convolutional layers
def generate_gradcam(model, image, predicted_class_index):
    conv_layers = ['conv2d_1', 'conv2d_2', 'conv2d_3'] # Define the convolution layers
    heatmaps = {}
    
    for layer_name in conv_layers:
        last_conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, predicted_class_index]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.math.reduce_max(heatmap)
        heatmaps[layer_name] = heatmap.numpy()
    
    return heatmaps

def show_heatmap_on_image(image, heatmaps, output_dir, base_filename):
    img = image[0]
    img = cv2.resize(img, (400, 400))
    img = np.uint8(255 * img)  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for layer_name, heatmap in heatmaps.items():
        heatmap_resized = cv2.resize(heatmap, (400, 400))
        heatmap_resized = np.uint8(255 * heatmap_resized)  # Ensure heatmap is in 0-255 range
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.5, heatmap_colored, 0.7, 0)
        output_path = os.path.join(output_dir, f"{base_filename}_{layer_name}_gradcam.jpg")
        cv2.imwrite(output_path, superimposed_img)

# Path to the input image folder
input_folder = "datasets\gradCam_dataset"
output_folder = "output\gradCam_output"

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_image_path = os.path.join(input_folder, filename)
        
        try:
            # Predict the class of the input image
            prediction, predicted_class_index, preprocessed_image = predict_image(input_image_path)
            heatmaps = generate_gradcam(model, preprocessed_image, predicted_class_index)
            image_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            show_heatmap_on_image(preprocessed_image, heatmaps, image_output_folder, os.path.splitext(filename)[0])

            print(f"Processed image: {filename}")
            print("Predicted class:", prediction)
        
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred with image {filename}: {e}")