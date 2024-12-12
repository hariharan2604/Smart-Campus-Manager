import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.ops.coco_eval import CocoEvaluator  # Correct import
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import torchvision.transforms as T
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Faster R-CNN model
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.to(device)
faster_rcnn_model.eval()

# Target classes (e.g., "person" and "car")
target_classes = ["person", "car"]

# COCO class IDs mapping for filtering
coco_class_ids = {
    1: "person",
    3: "car",
    # Add more classes if needed
}

# Filter dataset by classes
def filter_annotations_by_classes(dataset, target_classes):
    target_class_ids = [key for key, value in coco_class_ids.items() if value in target_classes]
    subset_images = []
    for idx, annotation in enumerate(dataset.coco.loadAnns(dataset.ids)):
        if annotation['category_id'] in target_class_ids:
            subset_images.append(dataset.ids[idx])
    return subset_images

# Custom CocoDetection class to filter by classes
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_classes=None):
        super().__init__(root, annFile, transform)
        self.ids = filter_annotations_by_classes(self, target_classes)


# Initialize the dataset with a subset of classes (person and car)
dataset_root = 'datasets/coco/images/val2017'  # Path to the COCO validation images
annotation_file = 'datasets/coco/annotations/instances_val2017.json'  # Path to the COCO annotations (instances_val2017.json)

filtered_dataset = CustomCocoDetection(
    root=dataset_root, 
    annFile=annotation_file,
    transform=T.ToTensor(), 
    target_classes=target_classes
)

# Create a DataLoader for the filtered dataset
data_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=False)

# Prepare COCO Evaluator (mAP calculation)
coco_evaluator = CocoEvaluator(filtered_dataset.coco, iou_types=["bbox"])

# Iterate through the dataset and evaluate
for images, targets in data_loader:
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Perform inference (no gradient computation needed)
    with torch.no_grad():
        outputs = faster_rcnn_model(images)
    
    # Format outputs for COCO Evaluator
    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    coco_evaluator.update(res)

# After looping through all images, accumulate and summarize results
coco_evaluator.accumulate()
coco_evaluator.summarize()
