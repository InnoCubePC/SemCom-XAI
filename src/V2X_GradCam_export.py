from YOLOv8_Explainer import yolov8_heatmap, display_images

model = yolov8_heatmap(
    weight="yolo.pt",       # Yolo model 
        conf_threshold=0.5,  
        method = "GradCAM", 
        ratio=0.05,
        show_box=False,
        renormalize=False,
)

imagelist = model(
    img_path="your/image.jpg")

display_images(imagelist)