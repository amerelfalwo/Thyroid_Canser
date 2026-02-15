import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from .metrix import dice_coef, iou_metric, bce_dice_loss

MODEL_PATH = "models/segmentation_best.h5"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "iou_metric": iou_metric,
        "bce_dice_loss": bce_dice_loss
    }
)


def extract_tirads_features(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    circularity = 4 * np.pi * area / (perimeter**2)
    masked_pixels = img[mask == 255]
    mean_intensity = np.mean(masked_pixels)
    calcification_ratio = np.sum(masked_pixels > 200) / len(masked_pixels)
    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "echogenicity": mean_intensity,
        "calcification_ratio": calcification_ratio
    }

def calculate_tirads(features):
    if features is None:
        return None
    score = 0
    if features["aspect_ratio"] > 1: score += 3
    if features["circularity"] < 0.6: score += 2
    if features["echogenicity"] < 100: score += 2
    if features["calcification_ratio"] > 0.05: score += 3
    if score <= 2: return 2
    elif score <= 4: return 3
    elif score <= 6: return 4
    else: return 5

def segment_image(img_path, save_dir="results", threshold=0.6):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img, (256,256)) / 255.0
    img_input = np.expand_dims(img_resized, axis=(0,-1))
    mask_pred = model.predict(img_input)[0,:,:,0]
    mask = (mask_pred > threshold).astype(np.uint8)
    mask_full = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask_full > 0)
    if len(xs) == 0 or len(ys) == 0:
        bbox = None
        roi = None
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        roi = orig[y_min:y_max+1, x_min:x_max+1]
    overlay = orig.copy()
    overlay[...,0] = np.maximum(overlay[...,0], mask_full*255)
    blended = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)
    mask_path = os.path.join(save_dir, "mask.png")
    overlay_path = os.path.join(save_dir, "overlay.png")
    cv2.imwrite(mask_path, mask_full*255)
    cv2.imwrite(overlay_path, blended)
    roi_path = None
    if roi is not None:
        roi_name = os.path.join(save_dir, "roi.png")
        Image.fromarray(roi).save(roi_name)
        roi_path = roi_name
    features = extract_tirads_features(img_path, mask_path)
    tirads = calculate_tirads(features)
    
    
    return {
        
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "bbox": bbox,
        "roi_path": roi_path,
        "tirads_features": features,
        "tirads": tirads
    }
