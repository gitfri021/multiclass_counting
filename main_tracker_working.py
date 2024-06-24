import cv2
import os
import sys
import numpy as np
import torch

# Ensure the boxmot folder is in the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'boxmot'))

from ultralytics import YOLO
from boxmot.trackers.ocsort.ocsort import OCSort

# Define paths
video_path = 'videos/vid3.mp4'  # Update this path
output_folder = 'output'

# Get the output file name based on the input file name
input_file_name = os.path.basename(video_path)
input_file_name_no_ext = os.path.splitext(input_file_name)[0]
output_video_path = os.path.join(output_folder, f"{input_file_name_no_ext}_results.mp4")

# Create necessary folders
os.makedirs(output_folder, exist_ok=True)

# Check if GPU is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize YOLOv8 model and OC-SORT tracker
model = YOLO('yolov8l.pt').to(device)  # Ensure the correct YOLOv8 model path

# Set a lower confidence threshold for detection
model.conf = 0.75  # You can adjust this value as needed
tracker = OCSort()

# Dictionary to keep track of object counts
crossed_objects = {}
counted_objects = set()

# Colors for different classes before crossing the ROI
class_colors = {
    0: (0, 0, 255),  # Red
    1: (255, 0, 0),  # Blue
    2: (0, 255, 255),  # Yellow
    3: (255, 255, 0),  # Cyan
    4: (255, 0, 255),  # Magenta
    5: (0, 128, 255),  # Orange
    6: (128, 0, 128),  # Purple
    7: (255, 255, 255),  # White
}

def detect_objects(frame):
    results = model(frame)
    detections = []
    for result in results:
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
            print(f"Detected: {model.names[int(cls)]}, Confidence: {conf}")  # Debugging: print detected classes
    return np.array(detections) if detections else np.empty((0, 6))

def track_objects(frame, detections):
    # Ensure the detections are properly formatted and the frame is passed to the tracker
    return tracker.update(detections, frame)

def update_counts_and_color(obj_id, cls):
    crossed_objects[cls] = crossed_objects.get(cls, 0) + 1
    counted_objects.add(obj_id)
    print(f"Object ID {obj_id} of class {cls} crossed the ROI")
    return (0, 255, 0)  # Change to green

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        thickness = cv2.FILLED
    
    # Draw the straight lines
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw the arcs
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def process_video(video_path, output_video_path, direction='top2bottom', roi=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set ROI based on the direction and frame dimensions
    if roi is None:
        if direction in ['top2bottom', 'bottom2top']:
            roi = frame_height // 2
        elif direction in ['left2right', 'right2left']:
            roi = frame_width // 2

    # Calculate the dashboard width (20% of the frame width)
    dashboard_width = frame_width // 5
    combined_width = frame_width + dashboard_width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (combined_width, frame_height))

    frame_count = 0

    # Dictionary to keep track of object IDs and their last known positions
    object_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a combined frame for video and dashboard
        combined_frame = np.zeros((frame_height, combined_width, 3), dtype=np.uint8)
        combined_frame[:, :frame_width] = frame

        # Draw ROI line and display ROI info
        if direction in ['top2bottom', 'bottom2top']:
            cv2.line(combined_frame, (0, roi), (frame_width, roi), (0, 0, 255), 2)
            cv2.putText(combined_frame, f"ROI: {roi}", (10, roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif direction in ['left2right', 'right2left']:
            cv2.line(combined_frame, (roi, 0), (roi, frame_height), (0, 0, 255), 2)
            cv2.putText(combined_frame, f"ROI: {roi}", (roi + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Object Detection
        detections = detect_objects(frame)

        # Object Tracking
        tracked_objects = track_objects(frame, detections)

        # Debugging: print tracked objects
        print(f"Tracked Objects: {tracked_objects}")

        # Draw tracked objects and update counts
        current_object_ids = set()
        for obj in tracked_objects:
            try:
                x1, y1, x2, y2, obj_id, _, cls, _ = obj
                current_object_ids.add(obj_id)
                cls = int(cls) if cls is not None else -1
                label = model.names[cls] if cls in model.names else 'Unknown'
                
                # Calculate centroid
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                
                # Check if the object has crossed the ROI and update counts and colors
                color = class_colors.get(cls, (0, 255, 255))  # Default color: yellow
                if obj_id in object_positions:
                    if direction == 'top2bottom' and object_positions[obj_id] < roi < centroid_y:
                        color = update_counts_and_color(obj_id, cls)
                    elif direction == 'bottom2top' and object_positions[obj_id] > roi > centroid_y:
                        color = update_counts_and_color(obj_id, cls)
                    elif direction == 'left2right' and object_positions[obj_id] < roi < centroid_x:
                        color = update_counts_and_color(obj_id, cls)
                    elif direction == 'right2left' and object_positions[obj_id] > roi > centroid_x:
                        color = update_counts_and_color(obj_id, cls)
                if obj_id in counted_objects:
                    color = (0, 255, 0)  # Change to green if already counted

                # Draw bounding box and centroid
                draw_rounded_rectangle(combined_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, 10)
                cv2.circle(combined_frame, (centroid_x, centroid_y), 3, color, -1)  # Smaller centroid dot
                cv2.putText(combined_frame, f"{centroid_x},{centroid_y}", (centroid_x + 5, centroid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(combined_frame, f"{str(obj_id)}-{label}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update the object's last known position
                if direction in ['top2bottom', 'bottom2top']:
                    object_positions[obj_id] = centroid_y
                elif direction in ['left2right', 'right2left']:
                    object_positions[obj_id] = centroid_x

            except Exception as e:
                print(f"Error drawing tracked object: {e}")

        # Remove IDs of objects that are no longer being tracked
        for obj_id in list(object_positions.keys()):
            if obj_id not in current_object_ids:
                del object_positions[obj_id]

        # Display counts on the dashboard
        y_offset = 30
        for cls, count in crossed_objects.items():
            color = class_colors.get(cls, (0, 255, 255))  # Default color: yellow
            cv2.putText(combined_frame, f"{model.names[int(cls)]} count: {count}", (frame_width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 30

        # Write the combined frame to the output video
        out.write(combined_frame)

        # Display the frame
        # cv2.namedWindow("Live Inference", cv2.WINDOW_NORMAL)
        # cv2.imshow('Live Inference', combined_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can specify a custom ROI and direction here, or use default values
    process_video(video_path, output_video_path, direction='top2bottom', roi=None)
    print("Processing completed.")
