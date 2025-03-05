import tensorflow as tf
import os
import cv2
import numpy as np

#Load the testing data (Typical form)

# Define the path to the data folders
data_folder = "datasets\gradCam_ASL_dataset"
folder_names = ["A", "B", "C",...]

num_classes = len(folder_names)

# Define the image size and channels
img_size = 100
channels = 3
    
# Define a function to load the data and labels
def load_data_test():
    data = []
    labels = []
    for i, folder_name in enumerate(folder_names):
        folder_path = os.path.join(data_folder, folder_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)                          # Read the image
            img_resized = cv2.resize(img, (img_size, img_size)) # Resize the img
            # extra line
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR) # Convert the img to BGR
            img_scaled = img_resized / 255.0  # Scale pixel values between 0 and 1
            data.append(img_scaled)
            labels.append(i)
    # Shuffle the data and labels
    shuffled_indices = np.random.permutation(len(data))
    data = np.array(data)[shuffled_indices]
    labels = np.array(labels)[shuffled_indices]
    return data, labels


x_test, y_test = load_data_test()