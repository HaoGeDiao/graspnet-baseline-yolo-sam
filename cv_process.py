import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

#取前n的连通面积
def retain_topn_area(mask,n):
    # 连通域分析
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if retval <= 1:  # 无有效连通域
        return mask
    
    # 提取面积并排序
    areas = stats[1:, 4]
    sorted_indices = np.argsort(areas)[::-1]
    sorted_areas = areas[sorted_indices]
    
    # 计算累加阈值
    total = np.sum(sorted_areas)
    cumulative = np.cumsum(sorted_areas) / total
    cutoff = np.argmax(cumulative >= n)
    
    # 生成新掩膜
    new_mask = np.zeros_like(labels)
    valid_labels = sorted_indices[:cutoff+1] + 1
    for lbl in valid_labels:
        new_mask[labels == lbl] = 255
        
    return new_mask

def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'models/yolo_and_sam/mobile_sam.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def set_classes(model, target_class):
    """Set YOLO-World model to detect specific class"""
    model.set_classes([target_class])


def detect_objects(image_path, target_class=None):
    """
    Detect objects with YOLO-World
    Returns: (list of bboxes in xyxy format, detected classes list, visualization image)
    """
    model = YOLO("models/yolo_and_sam/yolov8s-world.pt")
    if target_class:
        set_classes(model, target_class)

    results = model.predict(image_path)
    boxes = results[0].boxes
    vis_img = results[0].plot()  # Get visualized detection results

    # Extract valid detections
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.25:  # Confidence threshold
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": results[0].names[box.cls.item()]
            })

    return valid_boxes, vis_img


def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def segment_image(image_path, output_mask='cv_try/mask1.png'):
    # User input for target class
    use_target_class = input("Detect specific class? (yes/no): ").lower() == 'yes'
    target_class = input("Enter class name: ").strip() if use_target_class else None

    # Detect objects
    detections, vis_img = detect_objects(image_path, target_class)
    cv2.imwrite('detection_visualization.jpg', vis_img)

    # Prepare SAM predictor
    predictor = choose_model()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)  # Set image for SAM

    if detections:
        # Auto-select highest confidence bbox
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # Manual point selection
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                cv2.destroyAllWindows()

        cv2.setMouseCallback('Select Object', click_handler)
        cv2.waitKey(0)

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # Save results
    if mask is not None:
        mask = retain_topn_area(mask,0.8)
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"Segmentation saved to {output_mask}")
    else:
        print("mask1")

    return mask


if __name__ == '__main__':
    segment_image('cv_try/color.png')