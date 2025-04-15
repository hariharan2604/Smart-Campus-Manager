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


class VideoStreamServer:
    def __init__(self, video_path, port, model_path="models/yolo11x.pt"):
        self.video_path = video_path
        self.port = port
        self.model = YOLO(model_path)
        self.server = WebsocketServer(host="127.0.0.1", port=self.port)
        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_client_left(self.client_left)

        # Initialize tracking details
        self.class_track_ids = defaultdict(set)
        self.target_classes = {
            0,
            2,
            7,
            5,
            3,
            1,
            15,
            16,
            17,
            18,
            19,
            20,
        }  # Person, Car, Truck, Bus, Motorbike, Bicycle
        self.clients_connected = True  # Set to True to force processing without client
        self.stop_event = Event()

    def new_client(self, client, server):
        print(f"New client connected on port {self.port}: {client}")
        self.clients_connected = True

    def client_left(self, client, server):
        print(f"Client disconnected from port {self.port}: {client}")
        if len(server.clients) == 0:
            self.clients_connected = False

    def send_frame_to_clients(self, frame, unique_counts):
        # Encode frame and counts to JSON and send to all connected clients
        message = {"frame": frame, "instantaneous_unique_counts": unique_counts}
        self.server.send_message_to_all(json.dumps(message))
        print(f"Sent frame to clients on port {self.port}")

    def start_video_processing(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        w, h, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )
        print(f"Processing video {self.video_path} at {fps} FPS")

        frame_skip = int(fps / 15) if fps > 15 else 1

        try:
            frame_count = 0

            while not self.stop_event.is_set():
                if not self.clients_connected:
                    time.sleep(0.1)
                    continue

                ret, im0 = cap.read()
                if not ret:
                    print(f"Video processing completed for {self.video_path}.")
                    break

                if im0.shape[1] != w or im0.shape[0] != h:
                    im0 = cv2.resize(im0, (w, h))

                if frame_count % frame_skip == 0:
                    annotator = Annotator(im0, line_width=2)
                    results = self.model.track(im0, persist=True)

                    if (
                        results[0].boxes.id is not None
                        and results[0].boxes.cls is not None
                    ):
                        bboxes = results[0].boxes.xyxy
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        class_indices = results[0].boxes.cls.int().cpu().tolist()

                        for bbox, track_id, class_idx in zip(
                            bboxes, track_ids, class_indices
                        ):
                            if class_idx in self.target_classes:
                                class_name = self.model.names[class_idx]
                                label = f"{class_name} {track_id}"
                                self.class_track_ids[class_name].add(track_id)
                                annotator.box_label(
                                    bbox, label, color=colors(track_id, True)
                                )

                    _, buffer = cv2.imencode(
                        ".jpg", im0, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    last_encoded_frame = base64.b64encode(buffer).decode("utf-8")

                    instantaneous_unique_counts = {
                        class_name: len(track_id_set)
                        for class_name, track_id_set in self.class_track_ids.items()
                    }

                    self.send_frame_to_clients(
                        last_encoded_frame, instantaneous_unique_counts
                    )

                frame_count += 1

                if (
                    cv2.getWindowProperty("YOLO Object Tracking", cv2.WND_PROP_VISIBLE)
                    >= 1
                ):
                    cv2.imshow(f"YOLO Object Tracking - {self.video_path}", im0)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()

        except Exception as e:
            print(
                f"An error occurred with video {self.video_path} on port {self.port}: {e}"
            )

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def start(self):
        server_thread = Thread(target=self.server.run_forever)
        processing_thread = Thread(target=self.start_video_processing)

        server_thread.start()
        processing_thread.start()

        return server_thread, processing_thread


if __name__ == "__main__":
    video_sources = [
        ("samples/video1.mp4", 4001),
        ("samples/video2.mp4", 4002),
        ("samples/video3.mp4", 4003),
        ("samples/video4.mp4", 4004),
    ]

    servers = []
    threads = []
    for video_path, port in video_sources:
        server = VideoStreamServer(video_path, port)
        servers.append(server)
        server_threads = server.start()
        threads.extend(server_threads)

    for thread in threads:
        thread.join()
    print("All servers stopped.")
