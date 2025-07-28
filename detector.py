import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import base64

@dataclass
class DetectionResult:
    """Data class for YOLO detection results"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_point: Tuple[int, int]    # center x, y
    area: float

class YOLODetector:
    # Similar object mappings for common categories
    SIMILAR_OBJECTS = {
        "car": ["car", "truck", "bus"],
        "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "person": ["person"],
        "animal": ["cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "pet": ["cat", "dog"],
        "furniture": ["chair", "couch", "bed", "dining table"],
        "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
        "screen": ["tv", "laptop", "cell phone"],
        "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
        "fruit": ["banana", "apple", "orange"],
        "sports": ["sports ball", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "frisbee", "skis", "snowboard"],
        "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "microwave", "oven", "toaster", "sink", "refrigerator"],
        "bag": ["backpack", "handbag", "suitcase"],
        "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
        "plant": ["potted plant"],
        "toy": ["teddy bear", "kite"],
        "clothing": ["tie"],
        "timepiece": ["clock"],
        "book": ["book"],
        "utensil": ["fork", "knife", "spoon"],
        "beverage": ["bottle", "wine glass", "cup"]
    }
    
    def __init__(self, model_path='yolo11l.pt'):
        """Initialize YOLO11 Large detector"""
        print("Loading YOLO11 Large model...")
        self.model = YOLO(model_path)
        print("Model loaded!")
        self.cap = None
        
    def process_detections(self, results, image_shape) -> List[DetectionResult]:
        """Convert YOLO results to structured detection data"""
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Calculate center point and area
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                area = float((x2 - x1) * (y2 - y1))
                
                detection = DetectionResult(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center_point=(center_x, center_y),
                    area=area
                )
                detections.append(detection)
        
        return detections
    
    def filter_detections_by_class(self, detections: List[DetectionResult], 
                                   target_classes: List[str]) -> List[DetectionResult]:
        """Filter detections to only include specified object classes"""
        # Convert target classes to lowercase for case-insensitive matching
        target_classes_lower = [cls.lower() for cls in target_classes]
        
        filtered = []
        for detection in detections:
            if detection.class_name.lower() in target_classes_lower:
                filtered.append(detection)
        
        return filtered
    
    def expand_target_classes(self, user_input: List[str]) -> List[str]:
        """Expand user input to include similar objects"""
        expanded = set()
        
        for item in user_input:
            item_lower = item.lower()
            
            # Add the exact item if it exists in YOLO classes
            for yolo_class in self.model.names.values():
                if item_lower == yolo_class.lower():
                    expanded.add(yolo_class)
            
            # Add similar objects if mapping exists
            if item_lower in self.SIMILAR_OBJECTS:
                for similar in self.SIMILAR_OBJECTS[item_lower]:
                    expanded.add(similar)
        
        return list(expanded)
    
    def get_available_classes(self) -> List[str]:
        """Get list of all object classes YOLO can detect"""
        try:
            if hasattr(self.model, 'names') and self.model.names:
                return list(self.model.names.values())
            else:
                # Fallback if model.names is not accessible
                return [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
                    # ... add more as needed
                ]
        except Exception as e:
            print(f"Error accessing model names: {e}")
            return ["person", "car", "dog", "cat"]  # Minimal fallback
    
    def get_similar_object_mappings(self) -> Dict[str, List[str]]:
        """Get the similar object mappings dictionary"""
        return self.SIMILAR_OBJECTS
    
    def detect_specific_objects(self, image_path: str, target_objects: List[str], 
                               conf_threshold: float = 0.3, include_similar: bool = True,
                               return_annotated_image: bool = True, fallback_to_all: bool = True) -> dict:
        """Detect only specific objects requested by user, with fallback to all objects"""
        
        try:
            # Expand target objects if requested
            if include_similar:
                target_classes = self.expand_target_classes(target_objects)
            else:
                # Only include exact matches from YOLO classes
                target_classes = []
                for obj in target_objects:
                    for yolo_class in self.model.names.values():
                        if obj.lower() == yolo_class.lower():
                            target_classes.append(yolo_class)
            
            # Read and process image
            frame = cv2.imread(image_path)
            if frame is None:
                return {"error": "Cannot read image", "detections": []}
            
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            all_detections = self.process_detections(results, frame.shape)
            
            # Filter to only requested objects
            filtered_detections = self.filter_detections_by_class(all_detections, target_classes)
            
            # FALLBACK: If no matches found but other objects exist, show all objects
            used_fallback = False
            if len(filtered_detections) == 0 and len(all_detections) > 0 and fallback_to_all:
                filtered_detections = all_detections
                used_fallback = True
            
            # Prepare response
            response = {
                "requested_objects": target_objects,
                "searched_classes": target_classes,
                "total_objects_found": len(all_detections),
                "matching_objects_found": len(filtered_detections),
                "used_fallback": used_fallback,
                "fallback_message": "No requested objects found. Showing all detected objects." if used_fallback else None,
                "image_dimensions": {"width": frame.shape[1], "height": frame.shape[0]},
                "detections": []
            }
            
            # Add detections (filtered or all)
            for i, detection in enumerate(filtered_detections):
                # Calculate all vertices for the bounding box
                x1, y1, x2, y2 = detection.bbox
                vertices = {
                    "top_left": {"x": x1, "y": y1},
                    "top_right": {"x": x2, "y": y1},
                    "bottom_left": {"x": x1, "y": y2},
                    "bottom_right": {"x": x2, "y": y2}
                }
                
                response["detections"].append({
                    "id": i,
                    "class_name": detection.class_name,
                    "confidence": float(detection.confidence),
                    "matched_request": not used_fallback,  # Flag if this matched the request
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "vertices": vertices,  # All four corner points
                    "center_point": {
                        "x": detection.center_point[0],
                        "y": detection.center_point[1]
                    },
                    "area": detection.area,
                    "dimensions": {
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                })
            
            # Add annotated image if requested and detections exist
            if return_annotated_image and len(filtered_detections) > 0:
                # Draw detections with different colors for fallback
                annotated = frame.copy()
                for detection in filtered_detections:
                    x1, y1, x2, y2 = detection.bbox
                    # Different color for fallback detections
                    color = (0, 165, 255) if used_fallback else (0, 255, 0)  # Orange for fallback, green for matches
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    # Draw label
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    cv2.putText(annotated, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add fallback message on image if used
                if used_fallback:
                    cv2.putText(annotated, "Showing all objects (no matches found)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Convert to base64 for web transmission
                _, buffer = cv2.imencode('.jpg', annotated)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                response["annotated_image_base64"] = img_base64
            
            return response
            
        except Exception as e:
            return {"error": str(e), "detections": []}
    
    def detect_from_base64(self, image_base64: str, target_objects: List[str],
                          conf_threshold: float = 0.3, include_similar: bool = True,
                          fallback_to_all: bool = True) -> dict:
        """Detect objects from base64 encoded image (for web API)"""
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Save temporarily
            temp_path = f"temp_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, img)
            
            # Run detection
            result = self.detect_specific_objects(
                temp_path, target_objects, conf_threshold, include_similar, 
                return_annotated_image=True, fallback_to_all=fallback_to_all
            )
            
            # Clean up
            os.remove(temp_path)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}", "detections": []}
    
    def export_detection_data(self, detection_result: dict, output_dir: str = "detection_exports") -> str:
        """Export detection results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        json_path = os.path.join(output_dir, f"detection_result_{timestamp}.json")
        
        # Save without base64 image
        export_data = detection_result.copy()
        if "annotated_image_base64" in export_data:
            del export_data["annotated_image_base64"]
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return json_path 