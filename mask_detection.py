# Example Code: You can test this model on colab or anywhere u want

# Fix OpenMP runtime issue and suppress warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Install necessary libraries

# Download the model from Hugging Face
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from IPython.display import Image, display
import cv2
import matplotlib.pyplot as plt

# Define repository and file path
repo_id = "krishnamishra8848/Face_Mask_Detection"
filename = "D:/huggingface_models/models--krishnamishra8848--Face_Mask_Detection/snapshots/a4f3a438dcf144717f4b88459219d297b747408b/best.pt"  # File name in your Hugging Face repo
CACHE_DIR = 'D:/huggingface_models'
# Download the model file
# model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=CACHE_DIR)
# print(f"Model downloaded to: {model_path}")

# Load the YOLOv8 model
model = YOLO(filename)

# Define image paths for processing
image_paths = ["1.png", "2.png", "3.png", "4.png"]

# Process each image
for img_path in image_paths:
    print(f"\nProcessing image: {img_path}")
    
    # Display the original image
    print("Original Image:")
    display(Image(filename=img_path))
    
    # Run inference on the image
    print("Running inference...")
    results = model.predict(source=img_path, conf=0.5)
    
    # Save and visualize the results
    print("Saving and displaying predictions...")
    for result in results:
        # Annotate the image with bounding boxes and labels
        annotated_image = result.plot()  
        
        # Convert annotated image to RGB for display with matplotlib
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image_rgb)
        plt.axis("off")
        plt.title(f"Prediction Results - {img_path}")
        plt.show()
        
        # Save the annotated image to current directory
        output_path = f"result_{img_path}"
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved result to: {output_path}")
