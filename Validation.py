import os
import csv
from ultralytics import YOLO
import torch
from torch.profiler import profile, ProfilerActivity

def val():
    # Load the model
    model_name='yolo11n'
    model = YOLO(f"models/{model_name}.pt")  # Load a custom YOLO model

    # Validate the model and get basic metrics
    metrics = model.val(data='coco8.yaml',save_json=True,name=f"{model_name}")
    
    # Get the validation results directory
    val_res_dir = metrics.save_dir  # Path to validation results directory

    # Extract mAP metrics
    mAP50_95 = metrics.box.map        # mAP50-95
    mAP50 = metrics.box.map50         # mAP50
    mAP75 = metrics.box.map75         # mAP75
    
    # Model parameters (in millions)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6  # Convert to millions

    # Profiling with torch.profiler (CUDA only)
    flops = None
    if torch.cuda.is_available():
        with profile(
            activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
            profile_memory=True,
            with_flops=True
        ) as p:
            model.predict(source='images/bus.jpg', device='cuda')  # Profile only CUDA activity
            
        # Calculate total CUDA time and FLOPs
        total_cuda_time = sum(evt.self_device_time_total for evt in p.key_averages())
        speed_t4_trt = total_cuda_time / 1e3  # Convert to milliseconds
        flops = sum(evt.flops for evt in p.key_averages() if evt.flops is not None)
    else:
        speed_t4_trt = "N/A - GPU not available"
        flops = "Profiling requires GPU"

    # Prepare CSV data
    headers = ["Model","mAP 50-95", "mAP 50", "mAP 75", "Parameters (M)", "Speed T4 TensorRT (ms)", "FLOPs (B)"]
    values = [model_name,mAP50_95, mAP50, mAP75, f"{params:.2f}M", speed_t4_trt, f"{flops if isinstance(flops, str) else flops / 1e9:.2f} B"]

    # Save metrics to a CSV file in the validation results directory
    csv_path = os.path.join(val_res_dir, "metrics_results.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(values)

    print(f"Metrics saved to {csv_path}")
    print(f"mAP 50-95: {mAP50_95}")
    print(f"mAP 50: {mAP50}")
    print(f"mAP 75: {mAP75}")
    print(f"Speed T4 TensorRT (ms): {speed_t4_trt}")
    print(f"Parameters (M): {params:.2f}M")
    print(f"FLOPs (B): {flops if isinstance(flops, str) else flops / 1e9:.2f} B")

if __name__ == "__main__":
    val()
