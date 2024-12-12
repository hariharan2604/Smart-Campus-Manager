import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import random

# Load Faster R-CNN model
faster_rcnn_model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
faster_rcnn_model.eval()

# Target classes for detection
target_classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}

# Function to run Faster R-CNN detection
def faster_rcnn_detection(image):
    # Pre-process the image for Faster R-CNN
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    with torch.no_grad():  # No gradients needed for inference
        outputs = faster_rcnn_model(image_tensor)[0]
    
    # Extract bounding boxes, labels, and scores
    bboxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    return bboxes, labels, scores

# Function to draw bounding boxes and labels with scores on the image
def draw_boxes(image, detections, target_classes=None):
    # Generate unique colors for each class
    colors = {class_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for class_id in target_classes.keys()}

    for bbox, label, score in detections:
        # Only draw if the label is in the target classes
        if label not in target_classes:
            continue

        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)

        # Use the class ID to assign a color
        color = colors.get(int(label), (0, 255, 0))  # Default to green if no color is assigned
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        box_thickness = 3  # Thicker bounding box for better visibility

        # Draw a filled rectangle for text background for better visibility
        text_label = f"{target_classes.get(int(label), f'Class {int(label)})')}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text_label, font, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, thickness=cv2.FILLED)

        # Draw the bounding box with thicker edges and slight transparency
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=box_thickness)

        # Add a drop shadow effect to the text (for better visibility over varying backgrounds)
        shadow_offset = 2
        cv2.putText(
            image,
            text_label,
            (x1 + shadow_offset, y1 - baseline + shadow_offset),
            font,
            font_scale,
            (0, 0, 0),  # Shadow color (black)
            font_thickness + 1
        )
        
        # Put the label text on top
        cv2.putText(
            image,
            text_label,
            (x1, y1 - baseline),
            font,
            font_scale,
            (255, 255, 255),  # White text color for better contrast
            font_thickness
        )

# Process image and save the result
def process_image(image_path):
    # Load the image here
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or could not be loaded: {image_path}")

    # Run Faster R-CNN detection
    bboxes, labels, scores = faster_rcnn_detection(image)

    # Filter out detections with low scores (e.g., below 0.5) and keep only the target classes
    detections = [(bbox, label, score) for bbox, label, score in zip(bboxes, labels, scores) 
                  if score > 0.5 and label in target_classes]

    # Draw bounding boxes and labels for target classes
    draw_boxes(image, detections, target_classes=target_classes)

    # Save and show results
    result_path = "results/faster_rcnn_target_classes.jpg"
    cv2.imwrite(result_path, image)
    cv2.imshow("Final Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_path

# Example usage
image_path = "images/people1.jpg"
process_image(image_path)
