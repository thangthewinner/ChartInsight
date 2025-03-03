"""
Module for extracting bar chart values and calculating scores.
"""

import re
import numpy as np
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Định nghĩa đường dẫn
IMAGE_DIR = "./../assets/dataset/reduced_data/bardata(1031)/bar/images/test2019"
LABEL_DIR = "./../assets/dataset/reduced_data/bardata(1031)/bar/labels/test2019"
OBJECT_DETECTION_MODEL_DIR = "./../training/object_detection/runs/detect/train/weights/best.pt"
BAR_DETECTION_DIR = "./../testing/barchart/runs/detect/train/weights/best.pt"

# Load models
od_model = YOLO(OBJECT_DETECTION_MODEL_DIR)
bd_model = YOLO(BAR_DETECTION_DIR)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def detect_objects(model, image, class_id):
    """
    Detect objects in an image using a YOLO model.

    Args:
        model: YOLO model.
        image: Path or image array.
        class_id: Class ID to filter.

    Returns:
        np.ndarray: Bounding box of the detected object.
    """
    results = model(image)[0].boxes.data.cpu().numpy()
    filtered = results[results[:, 5] == class_id]
    return filtered[0] if len(filtered) > 0 else None

def get_plot_area(image_path):
    """Extract the plot area from an image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_area = detect_objects(od_model, image_path, class_id=4)
    return image[int(plot_area[1]):int(plot_area[3]), int(plot_area[0]):int(plot_area[2])] if plot_area is not None else None

def get_bar_anns(image):
    """Detect bars in the plot area."""
    return bd_model(image)[0].boxes.data.cpu().numpy()

def get_ocr_results(image):
    """Extract text using OCR."""
    return ocr_model.ocr(image, cls=True)

def is_numeric(text):
    """Check if a string is numeric."""
    return bool(re.match(r'^\d+(\.\d+)?$', re.sub(r'[^\d.]', '', text)))

def filter_axis_labels(ocr_results, threshold, axis='x'):
    """
    Filter OCR results for axis labels.

    Args:
        ocr_results: OCR results.
        threshold: Threshold coordinate for filtering.
        axis: 'x' for X-axis labels, 'y' for Y-axis labels.

    Returns:
        list: Filtered labels.
    """
    labels = []
    for res in ocr_results[0]:
        bbox, (text, conf) = res
        x_mean = np.mean([pt[0] for pt in bbox])
        y_mean = np.mean([pt[1] for pt in bbox])
        condition = (axis == 'x' and y_mean > threshold and not is_numeric(text)) or \
                    (axis == 'y' and x_mean < threshold and is_numeric(text))
        if condition:
            labels.append({'text': text, 'conf': conf, 'x_mean': x_mean, 'y_mean': y_mean})
    return labels

def merge_x_labels(x_labels, y_threshold=20, x_threshold=10):
    """
    Merge X-axis labels that belong to the same category.

    Args:
        x_labels (list): List of detected X-axis labels.
        y_threshold (int): Max vertical distance to merge labels.
        x_threshold (int): Max horizontal distance to consider as same category.

    Returns:
        list: Merged X-axis labels.
    """
    # Sắp xếp nhãn theo x_mean trước khi xử lý
    x_labels_sorted = sorted(x_labels, key=lambda lbl: (lbl['x_mean'], lbl['y_mean']))

    merged_labels = []
    used_indices = set()

    for i, lbl in enumerate(x_labels_sorted):
        if i in used_indices:
            continue

        full_text = lbl['text']
        closest_idx = None
        min_y_dist = float('inf')

        # Tìm nhãn gần nhất có cùng x_mean (hoặc gần nhau)
        for j in range(i + 1, len(x_labels_sorted)):
            if j in used_indices:
                continue
            
            other_lbl = x_labels_sorted[j]
            
            # Kiểm tra khoảng cách ngang (x_mean gần nhau)
            if abs(lbl['x_mean'] - other_lbl['x_mean']) < x_threshold:
                y_dist = abs(lbl['y_mean'] - other_lbl['y_mean'])
                
                # Nếu y_mean cách nhau hợp lý (1 trên, 1 dưới), ghép lại
                if y_dist < y_threshold and y_dist < min_y_dist:
                    min_y_dist = y_dist
                    closest_idx = j

        # Nếu tìm thấy nhãn phù hợp để ghép
        if closest_idx is not None:
            full_text = f"{x_labels_sorted[closest_idx]['text']} {full_text}"
            used_indices.add(closest_idx)

        merged_labels.append({
            'text': full_text,
            'conf': lbl['conf'],
            'x_mean': lbl['x_mean'],
            'y_mean': lbl['y_mean']
        })

    return merged_labels

def parse_number(text):
    """Convert text to a numeric value."""
    return float(re.sub(r'[^\d.]', '', text))

def get_bar_values(bars, x_labels, y_labels):
    """
    Assign labels to bars and calculate their values correctly.

    Args:
        bars (list): List of detected bars.
        x_labels (list): List of X-axis labels.
        y_labels (list): List of Y-axis labels.

    Returns:
        list: List of dictionaries containing bar bbox, label, and value.
    """
    if len(y_labels) < 2 or len(x_labels) < 2:
        return []

    # **Bước 1: Sắp xếp nhãn X và cột theo tọa độ x_mean**
    y_labels_sorted = sorted(y_labels, key=lambda lbl: lbl['y_mean'])
    x_labels_sorted = sorted(x_labels, key=lambda lbl: lbl['x_mean'])
    bars_sorted = sorted(bars, key=lambda bar: np.mean(bar[:2]))  # Lấy trung bình của x1 và x2

    # **Bước 2: Tính toán tỷ lệ y_scale**
    y_max, y_min = parse_number(y_labels_sorted[0]['text']), parse_number(y_labels_sorted[-1]['text'])
    plot_bottom = y_labels_sorted[-1]['y_mean']  # Đáy biểu đồ
    y_scale = (y_max - y_min) / (y_labels_sorted[-1]['y_mean'] - y_labels_sorted[0]['y_mean'])

    bar_data = []

    # **Bước 3: Gán nhãn X cho từng cột theo đúng thứ tự**
    for bar, x_label in zip(bars_sorted, x_labels_sorted):
        bar_value = (plot_bottom - bar[1]) * y_scale

        bar_data.append({
            'bbox': bar[:4],
            'label': x_label['text'],  # Gán nhãn đúng thứ tự
            'value': bar_value
        })

    return bar_data

