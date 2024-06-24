---

# 🚀 YOLOv8 & OC-SORT Object Tracking with ROI Counting 🖥️

Welcome to the **YOLOv8 & OC-SORT Object Tracking with ROI Counting** project! This script combines the power of YOLOv8 for object detection and OC-SORT for tracking, enhanced with unique features like rounded bounding boxes, color change on crossing a Region of Interest (ROI), and a live dashboard displaying counts of tracked objects. 

## 📋 Table of Contents
- [Features](#features)
- [Functionality](#functionality)
- [How to Use](#how-to-use)
- [Setting Up the Environment](#setting-up-the-environment)
- [Error Handling](#error-handling)
- [License](#license)

## 🌟 Features
- **Rounded Bounding Boxes**: Display objects with aesthetically pleasing rounded corners.
- **Color Change on Crossing ROI**: Track and visualize objects changing color once they cross a defined ROI.
- **Live Dashboard**: A side-by-side dashboard displaying counts of tracked objects by class.
- **Multiple Directions**: Configure tracking for objects moving in various directions (top to bottom, left to right, etc.).
- **Centroid Display**: Display the centroid of each bounding box with coordinates.

## 🛠️ Functionality
This script processes video inputs to detect and track objects, updating counts when objects cross a predefined ROI. Here's how it works, step-by-step:

1. **Initialization**: Set up paths, device configuration, and initialize the YOLOv8 model and OC-SORT tracker.
2. **Detection**: Use YOLOv8 to detect objects in each frame.
3. **Tracking**: Track detected objects across frames using OC-SORT.
4. **ROI Crossing Detection**: Check if objects have crossed the ROI and update their count.
5. **Drawing**: Draw rounded bounding boxes and centroids, and update colors upon crossing the ROI.
6. **Dashboard**: Display a live dashboard showing the counts of different classes of objects.
7. **Video Output**: Save the processed video with bounding boxes and dashboard.

## 📝 How to Use

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Set Up the Environment
Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install Required Libraries
Install the required libraries:
```bash
pip install torch opencv-python-headless ultralytics
```

### Step 4: Run the Script
Update the video path and output folder as needed in the script. Then, run the script:
```bash
python main.py
```

## 🛠️ Setting Up the Environment

1. **Python Installation**: Ensure Python 3.8 or higher is installed.
2. **Virtual Environment**: Create a virtual environment for the project:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Dependencies**: Install dependencies:
    ```bash
    pip install torch opencv-python-headless ultralytics
    ```

## ❗ Error Handling

### Error: `division by zero`
**Solution**: Ensure the bounding box coordinates are correct and not zero.

### Error: `list index out of range`
**Solution**: Ensure that detections are properly formatted and passed to the tracker.

### Slow Performance
**Solution**: Optimize the code by ensuring objects are removed from tracking once they leave the frame.

### Bounding Box Color Not Changing
**Solution**: Ensure the ROI condition and centroid calculations are correctly implemented.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Enjoy tracking with style and precision! 😎🚀 If you encounter any issues or have suggestions, feel free to open an issue or submit a pull request.

---
