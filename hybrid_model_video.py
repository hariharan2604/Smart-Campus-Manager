import cv2
import base64
import json
import torch
import numpy as np
from threading import Thread, Event
from ultralytics import YOLO
from websocket_server import WebsocketServer
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class HybridDetectionModel:
    def __init__(self):
        # Load YOLOv11 and Faster R-CNN models
        self.yolo_model = YOLO("models/yolo11x-seg.pt")  # Adjust path to your YOLOv11Seg model
        self.faster_rcnn_model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
        self.faster_rcnn_model.eval()

        # YOLO classes (filter only relevant ones)
        self.yolo_classes = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            5: "bus", 6: "train", 7: "truck"
        }
        self.selected_classes = {0, 1, 2, 3, 5, 6, 7}

    def yolo_detection(self, image):
        results = self.yolo_model(image)
        if isinstance(results, list) and len(results) > 0:
            results = results[0]
        
        if hasattr(results, "boxes"):
            bboxes = results.boxes.xyxy.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()

            filtered_bboxes, filtered_labels, filtered_scores = [], [], []
            for bbox, label, score in zip(bboxes, labels, scores):
                if label in self.selected_classes:
                    filtered_bboxes.append(bbox)
                    filtered_labels.append(label)
                    filtered_scores.append(score)

            return np.array(filtered_bboxes), np.array(filtered_labels), np.array(filtered_scores)
        else:
            raise ValueError("Unexpected results format")

    def faster_rcnn_refinement(self, image, bboxes):
        refined_detections = []
        for bbox in bboxes:
            roi = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            roi_tensor = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            outputs = self.faster_rcnn_model(roi_tensor)[0]
            for i in range(len(outputs["boxes"])):
                refined_box = outputs["boxes"][i].detach().numpy()
                score = outputs["scores"][i].item()
                label = outputs["labels"][i].item()
                refined_detections.append((refined_box, label, score))

        return refined_detections

    def merge_detections(self, yolo_results, refined_detections, iou_threshold=0.5):
        final_detections = []
        for yolo_box, yolo_label, yolo_score in zip(*yolo_results):
            best_iou = 0
            best_refined_detection = None
            for refined_box, refined_label, refined_score in refined_detections:
                iou = self.calculate_iou(yolo_box, refined_box)
                if iou > iou_threshold and refined_score > yolo_score:
                    if iou > best_iou:
                        best_iou = iou
                        best_refined_detection = (refined_box, refined_label, refined_score)

            if best_refined_detection:
                final_detections.append(best_refined_detection)
            else:
                final_detections.append((yolo_box, yolo_label, yolo_score))

        return final_detections

    def calculate_iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        return interArea / float(box1Area + box2Area - interArea)

    def hybrid_detection(self, image):
        yolo_results = self.yolo_detection(image)
        refined_detections = self.faster_rcnn_refinement(image, yolo_results[0])
        final_results = self.merge_detections(yolo_results, refined_detections)

        # Count detected classes
        class_counts = {}
        for _, label, _ in final_results:
            class_name = self.yolo_classes.get(int(label), f"Class {int(label)}")
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        return final_results, class_counts

    def draw_boxes(self, image, detections):
        for bbox, label, score in detections:
            x1, y1, x2, y2 = map(int, bbox)
            text_label = f"{self.yolo_classes.get(int(label), f'Class {int(label)})')}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


class VideoStreamServer:
    def __init__(self, video_path, port, detection_model):
        self.video_path = video_path
        self.port = port
        self.detection_model = detection_model
        self.server = WebsocketServer(host="127.0.0.1", port=self.port)
        self.clients_connected = False
        self.stop_event = Event()

        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_client_left(self.client_left)

    def new_client(self, client, server):
        print(f"New client connected to server {self.port}: {client}")
        self.clients_connected = True

    def client_left(self, client, server):
        print(f"Client disconnected from server {self.port}: {client}")
        if len(server.clients) == 0:
            self.clients_connected = False

    def send_frame_to_clients(self, frame, unique_counts):
        # Encode frame as base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        message = {
            "frame": encoded_image,
            "instantaneous_unique_counts": unique_counts
        }
        self.server.send_message_to_all(json.dumps(message))

    def start_video_processing(self):
        cap = cv2.VideoCapture(self.video_path)
        w, h, fps = (
            int(cap.get(x))
            for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
        )

        try:
            while not self.stop_event.is_set():
                if not self.clients_connected:
                    continue

                ret, im0 = cap.read()
                if not ret:
                    print(f"Video processing for source {self.port} completed.")
                    break

                if im0.shape[1] != w or im0.shape[0] != h:
                    im0 = cv2.resize(im0, (w, h))

                # Perform hybrid detection
                final_results, class_counts = self.detection_model.hybrid_detection(im0)
                self.detection_model.draw_boxes(im0, final_results)

                # Send frame and class counts to all clients
                self.send_frame_to_clients(im0, class_counts)

                # Show frame
                cv2.imshow(f"Stream {self.port}", im0)

                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.stop_event.set()
        self.server.shutdown()


class ServerManager:
    def __init__(self, video_sources, start_port=12345):
        self.video_sources = video_sources
        self.start_port = start_port
        self.detection_model = HybridDetectionModel()

    def start_server_for_video(self, video_path, port):
        server = VideoStreamServer(video_path, port, self.detection_model)
        server.start_video_processing()

    def run_multiple_servers(self):
        threads = []
        for idx, video_path in enumerate(self.video_sources):
            port = self.start_port + idx  # Assign different port for each video source
            thread = Thread(target=self.start_server_for_video, args=(video_path, port))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


# Example usage
video_sources = ["video1.mp4", "video2.mp4", "video3.mp4"]
server_manager = ServerManager(video_sources)
server_manager.run_multiple_servers()
