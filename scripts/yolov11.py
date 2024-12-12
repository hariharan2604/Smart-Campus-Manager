import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
from collections import Counter

# Load the YOLO model (replace with your correct path)
model = YOLO("models/yolo11x.pt")
# print(model.info())
IMG_PATH = 'images/people1.jpg'

# Perform inference on the image
validation_results = model(IMG_PATH)[0]

# Get the detections from the results
detections = sv.Detections.from_ultralytics(validation_results)

# Define the class IDs for people and vehicles (adjust if using a different dataset)
people_class_id = 0  # Person
vehicle_class_ids = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# Combine the class IDs into a single list
desired_class_ids = [people_class_id] + vehicle_class_ids

# Define class name mappings (adjust if using a different dataset)
class_names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Define colors for each class (BGR format for OpenCV)
class_colors = {
    0: (255, 0, 0),    # Blue for "person"
    2: (0, 255, 0),    # Green for "car"
    3: (0, 0, 255),    # Red for "motorcycle"
    5: (255, 255, 0),  # Cyan for "bus"
    7: (255, 0, 255)   # Magenta for "truck"
}

# Use numpy's isin to filter detections based on class ID
mask = np.isin(detections.class_id, desired_class_ids)
filtered_detections = sv.Detections(
    xyxy=detections.xyxy[mask],
    confidence=detections.confidence[mask],
    class_id=detections.class_id[mask]
)

# Count the number of occurrences of each class
class_counts = Counter(filtered_detections.class_id)

# Print the class counts
for class_id, count in class_counts.items():
    class_name = class_names.get(class_id, "Unknown")
    print(f"{class_name}: {count}")

# Load the image using OpenCV
img = cv2.imread(IMG_PATH)

# Draw bounding boxes and labels
for i in range(len(filtered_detections.xyxy)):
    # Get the bounding box coordinates
    x1, y1, x2, y2 = filtered_detections.xyxy[i].astype(int)
    
    # Get the class ID and class name
    class_id = filtered_detections.class_id[i]
    class_name = class_names.get(class_id, "Unknown")
    
    # Get the color for the class
    color = class_colors.get(class_id, (255, 255, 255))  # White if class not found
    
    # Draw the bounding box with the color corresponding to the class
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Annotate the class name above the bounding box
    cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save the annotated image to the 'results' directory
output_path = "results/annotated_yolov8_people1.jpg"
cv2.imwrite(output_path, img)

print(f"Annotated image saved at: {output_path}")
