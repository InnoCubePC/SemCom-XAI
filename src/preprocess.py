import cv2

# Function to preprocess the input image
def preprocess_image(image):
    img_size = 100
    channels = 3
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    image = image.reshape(1, img_size, img_size, channels)
    image = image / 255.0
    return image