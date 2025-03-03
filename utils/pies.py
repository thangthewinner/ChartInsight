""" 
Module for processing pie chart images and calculating the score between ground truth and prediction.
"""

import os
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform, cdist 
from ultralytics import YOLO

IMAGE_DIR = "./../assets/dataset/reduced_data/piedata(1008)/pie/images/test2019/"
LABEL_DIR = "./../assets/dataset/reduced_data/piedata(1008)/pie/labels/test2019/"
MODEL_DIR = "./../training/segmentation/runs/segment/train2/weights/best.pt"

# Load YOLO model
model = YOLO(MODEL_DIR)

def get_masks(image_path: str) -> np.ndarray:
    """
    Get predicted masks from the image by using YOLO segmentation model. 
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: List of masks in the image, each mask format (H, W).
    """
    results = model(image_path)[0]
    
    if not results.masks:  
        return np.array([])
    
    masks = results.masks.data 
    masks = masks.cpu().numpy()
    return masks

def get_gt_keypoints(image_path: str) -> np.ndarray:
    """
    Get ground truth bounding boxes from .txt file in label_path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: List of bounding boxes in the image, each bbox format [[x1, y1], [x2, y2], [xc, yc]].
    """
    image_name = os.path.basename(image_path)
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxs = []
    for line in lines:
        cat, x1, y1, x2, y2, xc, yc = map(float, line.split())
        bboxs.append([[x1, y1], [x2, y2], [xc, yc]])

    return np.array(bboxs)

def get_keypoint(mask: np.ndarray) -> np.ndarray:
    """
    Get 3 keypoint from mask by finding the 2 farthest points and the farthest point to the closest point between the 2 farthest points

    Args:
        mask (np.ndarray): mask in masks result from YOLO segmentation

    Returns:
        np.ndarray: 3 keypoint [[x1, y1], [x2, y2], [x3, y3]]
    """
    binary_mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0: 
        return 
    
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()

    dist_matrix = squareform(pdist(contour))

    idx1, idx2 = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    idx3 = np.argmax(np.min(dist_matrix[[idx1, idx2]], axis=0))

    keypoint = contour[[idx1, idx2, idx3]]

    return keypoint


def get_keypoints(masks: np.ndarray) -> np.ndarray:
    """
    Get keypoints of triangle from masks, each keypoint box format [[x1, y1], [x2, y2], [xc, yc]] (can be shuflled but keep the x, y order)
    Args:
        masks (np.ndarray): masks result from YOLO segmentation

    Returns:
        np.ndarray: shape (n, 6) with n is the number of mask segmented
    """
    keypoints = []
    for mask in masks:
        keypoint = get_keypoint(mask)
        if keypoint is None:
            continue
        keypoints.append(keypoint)
        
    return np.array(keypoints)
    

def get_triangle_flag(keypoints: np.ndarray) -> np.ndarray:
    """
    Get the boolean flag of the points in the triangle whether it is the center or not
    Args:
        keypoints (np.ndarray): keypoints of the triangle
    Returns:
        np.ndarray: flag bool shape (n, 3) with n is the number of keypoints, 3 is the number of points in the triangle
    """
    
    # case there're more than 3 keypoints
    if keypoints.shape[0] >= 3:
        t1, t2, t3 = keypoints[:3]

        # find the 2 closest pairs of t1 and t2
        dis_12 = cdist(t1, t2, metric="euclidean")

        _ = dis_12.ravel()

        # get the position of the first and second smallest
        pos1, pos2 = np.argsort(_)[:2]

        # unravel it to get the index of the points
        idx1 = np.unravel_index(pos1, dis_12.shape)
        idx2 = np.unravel_index(pos2, dis_12.shape)

        # get the pairs
        idx1_x, idx1_y = idx1
        pairs1 = t1[idx1_x], t2[idx1_y]

        idx2_x, idx2_y = idx2
        pairs2 = t1[idx2_x], t2[idx2_y]

        # compare with the third triangle to find the closest point
        dis_3 = cdist(t3, np.vstack([pairs1, pairs2]), metric="euclidean")
        idx = np.unravel_index(dis_3.argmin(), dis_3.shape)

        # IT IS THE CENTER!!!!
        point = t3[idx[0]]

    flag = np.zeros((keypoints.shape[0], 3))
    # find the closest point to the center for each triangle
    for i in range(keypoints.shape[0]):
        t = keypoints[i]
        dis = cdist(t, point.reshape(1, -1), metric="euclidean")
        idx = np.unravel_index(dis.argmin(), dis.shape)
        flag[i, idx[0]] = 1
        
    # case there're less than 3 keypoints
    else: 
        pass 
    
    
    return flag

def get_bboxs(masks: np.ndarray) -> np.ndarray:
    """
    Get the sorted bounding box of the masks, each bbox format [[x1, y1], [x2, y2], [xc, yc]]
    Args:
        masks (np.ndarray): masks result from YOLO segmentation

    Returns:
        np.ndarray: shape (n, 3, 2) with n is the number of mask segmented
    """
    keypoints = get_keypoints(masks)
    flag = get_triangle_flag(keypoints)

    sorted_triangles = np.array(
        [tri[np.argsort(f)] for tri, f in zip(keypoints, flag)]  # flag 1 go end
    )

    return sorted_triangles


def get_angle(bbox: np.ndarray) -> np.ndarray:
    """
    Get the angle BAC of the triangle bounding box in degree
    Args:
        bbox (np.ndarray): bounding box of the mask, format [[xA, yA], [xB, yB], [xc, yc]]

    Returns:
        np.ndarray: angle of the bounding box
    """

    vector_CA = bbox[2] - bbox[0]
    vector_CB = bbox[2] - bbox[1]

    dot_product = np.dot(vector_CA, vector_CB)

    norm_CA = np.linalg.norm(vector_CA)
    norm_CB = np.linalg.norm(vector_CB)

    cos_theta = dot_product / (norm_CA * norm_CB)
    angle = np.arccos(cos_theta)

    return np.degrees(angle)


def get_percent(bbox: np.ndarray) -> np.ndarray:
    """
    Get the percent of the arc in the pie chart.
    Args:
        bbox (np.ndarray): bounding box of the mask, format [xA, yA, xB, yB, xc, yc]

    Returns:
        np.ndarray: percent of the bounding box
    """
    angles = np.array([get_angle(b) for b in bbox])
    return angles / 360


def compute_score(ground_truth: np.ndarray, preds: np.ndarray) -> float:
    """
    Compute the score between ground truth and prediction of pie chart.
    See more in the document.

    Args:
        ground_truth (np.ndarray): ground truth percentages of the pie chart.
        preds (np.ndarray): predicted percentages of the pie chart.

    Returns:
        float: score between ground truth and prediction.
    """

    m, n = len(ground_truth), len(preds)
    score = np.zeros((m + 1, n + 1))  # score matrix

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # compute the match score
            match_score = 1 - abs(ground_truth[i - 1] - preds[j - 1]) / preds[j - 1]
            score[i, j] = max(
                score[i - 1, j], score[i, j - 1], score[i - 1, j - 1] + match_score
            )

    return score[-1, -1] / m

def get_score(image_path: str) -> float:
    """
    Get the score between ground truth and prediction of pie chart.
    Args:
        image_path (str): Path to the image file.

    Returns:
        float: score between ground truth and prediction.
    """
    gt_bboxs = get_gt_keypoints(image_path)
    gt_percent = get_percent(gt_bboxs)

    masks = get_masks(image_path)
    bboxs = get_bboxs(masks)
    percent = get_percent(bboxs)

    score = compute_score(gt_percent, percent)
    return score

def get_scores(image_dir: str) -> np.ndarray:
    """
    Get the scores between ground truth and prediction of pie chart in the image directory.
    Args:
        image_dir (str): Path to the image directory.

    Returns:
        np.ndarray: scores between ground truth and prediction.
    """
    image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    scores = []
    
    for image_path in image_paths:
        try:
            score = get_score(image_path)
            scores.append(score)
        except:
            scores.append(0)
    
    return np.array(scores)
    

if __name__ == "__main__":
    image_path = "./../assets/dataset/test/piedata(1008)/pie/images/test2019/f447ffede2ef85e73a191f8c1ed3f9df_c3RhdGxpbmtzLm9lY2Rjb2RlLm9yZwk5Mi4yNDMuMjMuMTM3.XLS-0-0.png"

    print("Ground truth percentage:")
    gt_bboxs = get_gt_keypoints(image_path)
    gt_percent = get_percent(gt_bboxs)
    print(sorted(gt_percent))
    
    print("Predicted percentage:")
    masks = get_masks(image_path)
    bboxs = get_bboxs(masks)
    percent = get_percent(bboxs)
    print(sorted(percent))

    score = compute_score(gt_percent, percent)
    print("Score:", score)
    
    scores = get_scores(IMAGE_DIR)
    print("Scores:", scores)