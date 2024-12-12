from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import json
from datetime import datetime

# Create the output directory if it doesn't exist
output_dir = "res"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Dictionary to store the unique track IDs for each class
class_track_ids = defaultdict(set)

# Load the YOLO model with segmentation capabilities
model = YOLO("models/yolo11x-seg.pt")

# Open the video file
video_path = "samples/video4.mp4"
cap = cv2.VideoCapture(video_path)

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Generate a unique filename using the current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get the original video name without extension
nested_output_dir = os.path.join(output_dir, video_name)
os.makedirs(nested_output_dir, exist_ok=True)  # Create nested output directory

output_filename = os.path.join(nested_output_dir, f"instance-segmentation-object-tracking_{timestamp}.mp4")

# Initialize video writer to save the output video as MP4 using the 'H264' codec
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"H264"), fps, (w, h))

# Define indices for classes to annotate
person_index = 0  # Adjust this if your model has a different index for 'person'
car_index = 2     # Adjust this index for 'car'
truck_index = 7   # Adjust this index for 'truck'
bus_index = 5     # Adjust this index for 'bus'
motorbike_index = 3  # Adjust this index for 'motorbike'
bicycle_index = 1  # Adjust this index for 'bicycle'

# Create a set of indices for vehicles and people
target_classes = {person_index, car_index, truck_index, bus_index, motorbike_index, bicycle_index}

# Create a named window with normal flag to allow resizing
cv2.namedWindow("instance-segmentation-object-tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("instance-segmentation-object-tracking", w, h)  # Resize the window to the video dimensions

while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        print("Failed to capture video frame or video processing has been successfully completed.")
        break

    # Ensure the frame retains the original size
    if im0.shape[1] != w or im0.shape[0] != h:
        im0 = cv2.resize(im0, (w, h))

    # Create an annotator object to draw on the frame
    annotator = Annotator(im0, line_width=2)

    # Perform object tracking on the current frame
    results = model.track(im0, persist=True)

    # Check if tracking IDs, class names, and masks are present in the results
    if results[0].boxes.id is not None and results[0].masks is not None and results[0].boxes.cls is not None:
        # Extract masks, tracking IDs, and class indices
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        # Annotate each mask with its corresponding tracking ID, class name, and color
        for mask, track_id, class_idx in zip(masks, track_ids, class_indices):
            # Check if the detected class is in our target classes
            if class_idx in target_classes:
                class_name = model.names[class_idx]  # Get the class name using the class index
                label = f"{class_name} {track_id}"  # Combine class name with track ID for the label

                # Add the track ID to the set of the corresponding class to ensure unique counts
                class_track_ids[class_name].add(track_id)

                # Annotate the frame with the mask, color, and label
                annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=label)

    # Write the annotated frame to the output video and check success
    success = out.write(im0)
    if not success:
        print("Error writing frame to video.")

    # Display the annotated frame
    cv2.imshow("instance-segmentation-object-tracking", im0)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video writer and capture objects, and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()

# Save the total unique count of each class (using unique track IDs) to a JSON file
result_filename = os.path.join(nested_output_dir, "instance-segmentation-results.json")
with open(result_filename, 'w') as json_file:
    json.dump({class_name: len(track_id_set) for class_name, track_id_set in class_track_ids.items()}, json_file, indent=4)

# Print the total unique count of each class (using unique track IDs)
print("\nTotal count of each class (based on unique track IDs):")
for class_name, track_id_set in class_track_ids.items():
    print(f"{class_name}: {len(track_id_set)}")
