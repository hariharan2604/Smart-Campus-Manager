import os
import cv2
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
import json

# Create the output directory if it doesn't exist
output_dir = "res"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store tracking history with default empty lists
class_track_ids = defaultdict(set)

# Load the YOLO detection model
model = YOLO("models/yolov8x.pt")  # Using detection model

# Open the video file
video_path = "samples/video4.mp4"
cap = cv2.VideoCapture(video_path)

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Generate a unique filename using the current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = os.path.splitext(os.path.basename(video_path))[0]
nested_output_dir = os.path.join(output_dir, video_name)
os.makedirs(nested_output_dir, exist_ok=True)

# Temporary output filename before ffmpeg conversion
output_filename = os.path.join(nested_output_dir, f"object-detection_{timestamp}.mp4")

# Initialize video writer to save the output video using 'mp4v'
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"h264"), fps, (w, h))

# Define class indices (person, car, etc.)
person_index = 0
car_index = 2
truck_index = 7
bus_index = 5
motorbike_index = 3
bicycle_index = 1
target_classes = {person_index, car_index, truck_index, bus_index, motorbike_index, bicycle_index}

cv2.namedWindow("object-detection-object-tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("object-detection-object-tracking", w, h)

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video processing completed.")
        break

    if im0.shape[1] != w or im0.shape[0] != h:
        im0 = cv2.resize(im0, (w, h))

    annotator = Annotator(im0, line_width=2)

    # Perform object detection with tracking on the current frame
    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].boxes.cls is not None:
        bboxes = results[0].boxes.xyxy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        for bbox, track_id, class_idx in zip(bboxes, track_ids, class_indices):
            if class_idx in target_classes:
                class_name = model.names[class_idx]
                label = f"{class_name} {track_id}"
                class_track_ids[class_name].add(track_id)
                annotator.box_label(bbox, label, color=colors(track_id, True))

    out.write(im0)
    cv2.imshow("object-detection-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()


# Save the tracking results to a JSON file
result_filename = os.path.join(nested_output_dir, "object-detection-results.json")
with open(result_filename, 'w') as json_file:
    json.dump({class_name: len(track_id_set) for class_name, track_id_set in class_track_ids.items()}, json_file, indent=4)

# Print the total unique count of each class
print("\nTotal count of each class (based on unique track IDs):")
for class_name, track_id_set in class_track_ids.items():
    print(f"{class_name}: {len(track_id_set)}")
