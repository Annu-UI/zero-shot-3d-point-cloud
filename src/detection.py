from ultralytics import YOLOWorld
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_size="yolov8s-worldv2"):
        """
        Initialize YOLO-World detector.
        model_size options: yolov8s-worldv2 (fast), yolov8l-worldv2 (accurate)
        """
        self.model = YOLOWorld(model_size)
        print(f"[Detector] Loaded {model_size}")

    def detect(self, image_path, classes, conf_threshold=0.3):
        """
        Detect objects in image.
        
        Args:
            image_path: path to image file
            classes: list of class names e.g. ["chair", "person"]
            conf_threshold: minimum confidence score
            
        Returns:
            boxes: list of [x1, y1, x2, y2] bounding boxes
            scores: list of confidence scores
            labels: list of class names
            image: original image as numpy array
        """
        # Set target classes dynamically — this is YOLO-World's key feature
        self.model.set_classes(classes)
        
        # Run inference
        results = self.model.predict(image_path, conf=conf_threshold)
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, scores, labels = [], [], []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                label = classes[int(box.cls[0])]
                
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(score)
                labels.append(label)
        
        print(f"[Detector] Found {len(boxes)} object(s): {labels}")
        return boxes, scores, labels, image


def visualize_detections(image, boxes, labels, scores):
    """Draw bounding boxes on image for verification."""
    vis = image.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis