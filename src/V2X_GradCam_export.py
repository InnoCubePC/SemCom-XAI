import os
import logging
from YOLOv8_Explainer import yolov8_heatmap, display_images
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class XAI_YOLOv8_Analyzer:
    """
    A class for generating explainability heatmaps using a YOLOv8 model.
    """

    def __init__(self, weight="yolo.pt", conf_threshold=0.5, methods=None, output_dir="outputs/"):
        """
        Initializes the XAI_YOLOv8_Analyzer with configuration parameters.

        :param weight: Path to the YOLO model weights (default: "yolo.pt").
        :param conf_threshold: Confidence threshold for predictions (default: 0.5).
        :param methods: List of explainability methods to apply (default: ["GradCAM"]).
        :param output_dir: Directory to save generated heatmaps.
        """
        self.weight = weight
        self.conf_threshold = conf_threshold
        self.methods = methods if methods else ["GradCAM", "GradCAM++", "ScoreCAM"]
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info("XAI_YOLOv8_Analyzer initialized with the following settings:")
        logging.info(f" - Model Weights: {self.weight}")
        logging.info(f" - Confidence Threshold: {self.conf_threshold}")
        logging.info(f" - Methods: {self.methods}")
        logging.info(f" - Output Directory: {self.output_dir}")

    def preprocess_image(self, image_path):
        """
        Preprocess an image before feeding it into the model.
        
        Steps:
        - Read image using OpenCV
        - Convert BGR to RGB (if needed)
        - Resize to a fixed shape (640x640)
        - Normalize pixel values (0 to 1)
        - Convert to a NumPy array

        :param image_path: Path to the input image.
        :return: Processed image as a NumPy array.
        """
        logging.info(f"Preprocessing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to (640, 640) – adjust based on your model’s input size
        target_size = (640, 640)
        image = cv2.resize(image, target_size)

        # Normalize pixel values to [0,1] range
        image = image.astype(np.float32) / 255.0

        # Add batch dimension (1, H, W, C) if required by your model
        image = np.expand_dims(image, axis=0)

        logging.info(f"Image preprocessed: shape={image.shape}, dtype={image.dtype}")
        
        return image

    def generate_heatmaps(self, image_paths):
        """
        Generates heatmaps for a list of images using different XAI methods.

        :param image_paths: List of image file paths.
        """
        for method in self.methods:
            logging.info(f"Applying {method} on {len(image_paths)} images...")

            # Initialize YOLO model with explainability method
            model = yolov8_heatmap(
                weight=self.weight,
                conf_threshold=self.conf_threshold,
                method=method,
                ratio=0.05,
                show_box=True,
                renormalize=True,
            )

            for img_path in image_paths:
                processed_img = self.preprocess_image(img_path)
                
                logging.info(f"Generating heatmap for: {processed_img}")
                heatmaps = model(img_path=processed_img)

                # Save heatmaps
                self.save_heatmaps(heatmaps, img_path, method)

                # Display results
                display_images(heatmaps)

    def save_heatmaps(self, heatmaps, original_img, method):
        """
        Saves generated heatmaps to the output directory.

        :param heatmaps: List of heatmap images.
        :param original_img: Original image path for naming.
        :param method: Explainability method used.
        """
        base_name = os.path.basename(original_img).split('.')[0]  # Extract filename
        save_path = os.path.join(self.output_dir, f"{base_name}_{method}.jpg")

        logging.info(f"Saving heatmap to {save_path}")
        # Placeholder: Replace with actual saving function (e.g., PIL.Image.save())
        # Example: heatmaps[0].save(save_path) if using PIL
        pass

if __name__ == "__main__":
    # Example image list (placeholders)
    image_list = ["your/image1.jpg", "your/image2.jpg", "your/image3.jpg"]

    # Initialize and run the analyzer
    xai_analyzer = XAI_YOLOv8_Analyzer()
    xai_analyzer.generate_heatmaps(image_list)
