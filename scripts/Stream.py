from threading import Thread, Event
import os
import cv2
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import base64
import json
from websocket_server import WebsocketServer
import time

# Prepare output directory
output_dir = "res"
os.makedirs(output_dir, exist_ok=True)

# Class tracking and target classes
class_track_ids = defaultdict(set)
person_index = 0
car_index = 2
truck_index = 7
bus_index = 5
motorbike_index = 3
bicycle_index = 1
target_classes = {person_index, car_index, truck_index, bus_index, motorbike_index, bicycle_index}

# Connection and stop events
clients_connected = False
stop_event = Event()

def new_client(client, server):
    global clients_connected
    print("New client connected:", client)
    clients_connected = True

def client_left(client, server):
    global clients_connected
    print("Client disconnected:", client)
    if len(server.clients) == 0:
        clients_connected = False

def send_frame_to_clients(server, frame, unique_counts):
    # Encode frame and counts to JSON and send to all connected clients
    message = {"frame": frame, "instantaneous_unique_counts": unique_counts}
    server.send_message_to_all(json.dumps(message))

def start_video_processing(server):
    global clients_connected
    model = YOLO("models/yolo11n.pt")  # Specify YOLO model path
    video_path = "samples/video2.mov"
    cap = cv2.VideoCapture(video_path)

    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    frame_skip = int(fps / 15) if fps > 15 else 1  # Limit to 15 FPS or less if original fps is lower

    try:
        frame_count = 0
        last_encoded_frame = None

        while not stop_event.is_set():
            if not clients_connected:
                time.sleep(0.1)
                continue  # Skip processing if no clients are connected

            ret, im0 = cap.read()
            if not ret:
                print("Video processing completed.")
                break

            if im0.shape[1] != w or im0.shape[0] != h:
                im0 = cv2.resize(im0, (w, h))

            # Skip frames if frame_count % frame_skip != 0
            if frame_count % frame_skip == 0:
                annotator = Annotator(im0, line_width=2)
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

                # Encode the frame as JPEG and cache the encoded frame
                _, buffer = cv2.imencode('.jpg', im0, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                last_encoded_frame = base64.b64encode(buffer).decode("utf-8")

                # Calculate instantaneous unique counts
                instantaneous_unique_counts = {
                    class_name: len(track_id_set)
                    for class_name, track_id_set in class_track_ids.items()
                }

                # Send frame and unique counts to all clients
                send_frame_to_clients(server, last_encoded_frame, instantaneous_unique_counts)

            frame_count += 1

            # Optional: Display frame in a local window for debugging
            cv2.imshow("YOLO Object Tracking", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()

    except Exception as e:
        print("An error occurred:", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start WebSocket server
    server = WebsocketServer(host="127.0.0.1", port=3001)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)

    # Start video processing in a separate thread
    print("Starting WebSocket server on port 3001...")
    server_thread = Thread(target=start_video_processing, args=(server,))
    server_thread.start()

    # Run the WebSocket server
    server.run_forever()

    # Wait for the video processing thread to finish
    server_thread.join()
    print("Stopping the server.")
