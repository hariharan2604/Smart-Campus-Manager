import torch
import cv2
import glob
import os
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import random

# Load YOLOv11 and Faster R-CNN models
yolo_model = YOLO("models/yolo11x-seg.pt")  # Adjust path to your YOLOv11Seg model
faster_rcnn_model = fasterrcnn_resnet50_fpn(
    weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT"
)
faster_rcnn_model.eval()
image_dir = "C:\\Projects\\Smart-Campus-Manager\\images"

image_paths = glob.glob(os.path.join(image_dir,"*"))  # Adjust extension if needed


# YOLOv11 class names (keep only relevant ones for this example)
yolo_classes = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}
selected_classes = {
    0,
    1,
    2,
    3,
    5,
    6,
    7,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
}  # Only person and vehicle classes


# Function to run YOLOv11 detection and filter selected classes
def yolo_detection(image):
    results = yolo_model(image)
    results[0].show()
    if isinstance(results, list) and len(results) > 0:
        results = results[0]

    if hasattr(results, "boxes"):
        bboxes = results.boxes.xyxy.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        # Filter results for selected classes
        filtered_bboxes, filtered_labels, filtered_scores = [], [], []
        for bbox, label, score in zip(bboxes, labels, scores):
            if label in selected_classes:
                filtered_bboxes.append(bbox)
                filtered_labels.append(label)
                filtered_scores.append(score)

        return (
            np.array(filtered_bboxes),
            np.array(filtered_labels),
            np.array(filtered_scores),
        )
    else:
        raise ValueError("Unexpected results format")


# Function to refine detections using Faster R-CNN
def faster_rcnn_refinement(image, bboxes):
    refined_detections = []
    for bbox in bboxes:
        roi = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        roi_tensor = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Get the outputs from Faster R-CNN
        outputs = faster_rcnn_model(roi_tensor)[0]
        for i in range(len(outputs["boxes"])):
            refined_box = outputs["boxes"][i].detach().numpy()
            score = outputs["scores"][i].item()
            label = outputs["labels"][i].item()
            refined_detections.append((refined_box, label, score))

    return refined_detections


def merge_detections(yolo_results, refined_detections, iou_threshold=0.5):
    final_detections = []

    for yolo_box, yolo_label, yolo_score in zip(*yolo_results):
        best_iou = 0  # Initialize the best IoU score as 0
        best_refined_detection = (
            None  # Initialize the variable to store the best matching refined detection
        )

        # Compare the current YOLO detection with each refined detection from Faster R-CNN
        for refined_box, refined_label, refined_score in refined_detections:
            iou = calculate_iou(
                yolo_box, refined_box
            )  # Calculate IoU between YOLO and refined box
            if iou > iou_threshold and refined_score > yolo_score:
                if iou > best_iou:  # If the IoU is better than the previous best IoU
                    best_iou = iou  # Update the best IoU
                    best_refined_detection = (
                        refined_box,
                        refined_label,
                        refined_score,
                    )  # Update best refined detection

        if best_refined_detection:
            final_detections.append(
                best_refined_detection
            )  # If a better refined detection exists, use it
        else:
            final_detections.append(
                (yolo_box, yolo_label, yolo_score)
            )  # Otherwise, use the original YOLO detection

    return final_detections


# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


# Hybrid detection pipeline
def hybrid_detection(image):
    yolo_results = yolo_detection(image)
    refined_detections = faster_rcnn_refinement(image, yolo_results[0])
    final_results = merge_detections(yolo_results, refined_detections)

    # Count detected classes
    class_counts = {}
    for _, label, _ in final_results:
        class_name = yolo_classes.get(int(label), f"Class {int(label)}")
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    return final_results, class_counts


# Process image and save the result
def process_image(image_path):
    # Load the image here
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or could not be loaded: {image_path}")

    final_results, class_counts = hybrid_detection(image)
    draw_boxes(image, final_results, yolo_classes=yolo_classes)

    # Save and show results
    result_path = f"results/{os.path.basename(image_path)}.jpg"
    cv2.imwrite(result_path, image)
    cv2.imshow("Final Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print class counts
    print("Detected classes and counts:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    return result_path


# Function to draw bounding boxes and labels with scores on the image
def draw_boxes(image, detections, yolo_classes=None):
    # Generate unique colors for each class
    colors = {
        class_id: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for class_id in yolo_classes.keys()
    }

    for bbox, label, score in detections:
        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)

        # Use the class ID to assign a color
        color = colors.get(
            int(label), (0, 255, 0)
        )  # Default to green if no color is assigned
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        box_thickness = 3  # Thicker bounding box for better visibility

        # Draw a filled rectangle for text background for better visibility
        text_label = (
            f"{yolo_classes.get(int(label), f'Class {int(label)})')}: {score:.2f}"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            text_label, font, font_scale, font_thickness
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color,
            thickness=cv2.FILLED,
        )

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
            font_thickness + 1,
        )

        # Put the label text on top
        cv2.putText(
            image,
            text_label,
            (x1, y1 - baseline),
            font,
            font_scale,
            (255, 255, 255),  # White text color for better contrast
            font_thickness,
        )


# Example usage
image_path = "images/RobBrandford_ExecutiveDirector-scaled.jpg"
# for image_path in image_paths:
    # Read image
print(image_path)
# img = cv2.imread(image_path)
process_image(image_path)
