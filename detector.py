import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import base64
import gc
import logging
import psutil
import torch

# Set up logging
logger = logging.getLogger(__name__)

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
    
    def __init__(self, model_path='yolo11l.pt'):  # Changed to Large model
        try:
            logger.info(f"Loading YOLO11 Large model from: {model_path}")
            
            # Railway-specific optimizations
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'  # Optimize for Railway's CPU
            
            # Check available memory before loading
            memory = psutil.virtual_memory()
            logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
            
            # FIXED: Properly detect and configure device for Railway
            self.device = self._get_optimal_device()
            logger.info(f"Using device: {self.device}")
            
            # Configure PyTorch for the detected device
            if self.device != 'cpu':
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            else:
                logger.info("Using CPU inference - optimizing for Railway")
                torch.set_num_threads(4)  # Optimize CPU usage for Railway
            
            # Load model with error handling
            self.model = YOLO(model_path)
            
            # Verify model loaded correctly
            if not hasattr(self.model, 'names') or not self.model.names:
                raise ValueError("Model loaded but class names are not accessible")
            
            logger.info(f"YOLO Large model loaded successfully! Available classes: {len(self.model.names)}")
            logger.info(f"Sample classes: {list(self.model.names.values())[:10]}")
            
            # Memory cleanup after loading
            gc.collect()
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            
            self.cap = None
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Cannot initialize YOLO detector: {e}")
    
    def _get_optimal_device(self):
        """Determine the best device for Railway deployment"""
        try:
            # Check if CUDA is available and working
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Test if we can actually use CUDA
                try:
                    torch.cuda.current_device()
                    return '0'  # Use first GPU
                except Exception as e:
                    logger.warning(f"CUDA available but not accessible: {e}")
                    return 'cpu'
            else:
                logger.info("CUDA not available, using CPU")
                return 'cpu'
        except Exception as e:
            logger.warning(f"Error detecting device: {e}, defaulting to CPU")
            return 'cpu'
    
    def get_memory_usage(self):
        """Get current memory usage for monitoring"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent()
        }
    
    def _check_image_memory_requirements(self, height: int, width: int) -> dict:
        """Check if image can be processed within memory limits"""
        # Calculate approximate memory needed for image processing
        # RGB image: height * width * 3 channels * 4 bytes (float32) * multiple copies during processing
        estimated_memory_mb = (height * width * 3 * 4 * 6) / (1024 * 1024)  # 6x for processing overhead
        
        current_memory = psutil.virtual_memory()
        available_mb = current_memory.available / (1024 * 1024)
        
        return {
            "estimated_memory_mb": estimated_memory_mb,
            "available_memory_mb": available_mb,
            "can_process": estimated_memory_mb < (available_mb * 0.8),  # Use max 80% of available memory
            "memory_ratio": estimated_memory_mb / available_mb
        }
        
    def process_detections(self, results, image_shape) -> List[DetectionResult]:
        """Convert YOLO results to structured detection data with original image coordinates"""
        detections = []
        
        try:
            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                logger.info(f"Processing {len(results[0].boxes)} detections from original image {image_shape}")
                
                for i, box in enumerate(results[0].boxes):
                    try:
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls in self.model.names:
                            class_name = self.model.names[cls]
                        else:
                            logger.warning(f"Unknown class index: {cls}")
                            continue
                            
                        # Get coordinates - these are already in original image coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Validate bounding box coordinates against original image dimensions
                        height, width = image_shape[:2]
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(x1 + 1, min(x2, width))
                        y2 = max(y1 + 1, min(y2, height))
                        
                        if x2 <= x1 or y2 <= y1:
                            logger.warning(f"Invalid bounding box after clipping: ({x1}, {y1}, {x2}, {y2})")
                            continue
                        
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
                        
                        logger.debug(f"Detection {i}: {class_name} ({conf:.2f}) at original coords ({x1}, {y1}, {x2}, {y2})")
                        
                    except Exception as e:
                        logger.error(f"Error processing detection {i}: {e}")
                        continue
            else:
                logger.info("No detections found in results")
        
        except Exception as e:
            logger.error(f"Error in process_detections: {e}")
        
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
                classes = list(self.model.names.values())
                logger.debug(f"Retrieved {len(classes)} classes from model")
                return classes
            else:
                logger.warning("Model.names not accessible, using fallback")
                # Fallback if model.names is not accessible
                return [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                ]
        except Exception as e:
            logger.error(f"Error accessing model names: {e}")
            return ["person", "car", "dog", "cat"]  # Minimal fallback
    
    def get_similar_object_mappings(self) -> Dict[str, List[str]]:
        """Get the similar object mappings dictionary"""
        return self.SIMILAR_OBJECTS
    
    def detect_specific_objects(self, image_path: str, target_objects: List[str], 
                               conf_threshold: float = 0.25, include_similar: bool = True,  # Lower threshold for Large model
                               return_annotated_image: bool = True, fallback_to_all: bool = True) -> dict:
        """Detect objects with EXACT original image coordinates - NO RESIZING"""
        
        try:
            logger.info(f"Starting detection for: {target_objects}")
            logger.info(f"Confidence threshold: {conf_threshold}")
            logger.info("PRESERVING ORIGINAL IMAGE DIMENSIONS - coordinates will be exact")
            
            # Log memory usage before detection
            memory_before = self.get_memory_usage()
            logger.info(f"Memory before detection: {memory_before}")
            
            # Expand target objects if requested
            if include_similar:
                target_classes = self.expand_target_classes(target_objects)
                logger.info(f"Expanded target classes: {target_classes}")
            else:
                # Only include exact matches from YOLO classes
                target_classes = []
                for obj in target_objects:
                    for yolo_class in self.model.names.values():
                        if obj.lower() == yolo_class.lower():
                            target_classes.append(yolo_class)
                logger.info(f"Exact match target classes: {target_classes}")
            
            # Read and process image with error handling
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}", "detections": []}
            
            original_frame = cv2.imread(image_path)
            if original_frame is None:
                return {"error": f"Cannot read image file: {image_path}", "detections": []}
            
            # Get original dimensions
            original_height, original_width = original_frame.shape[:2]
            logger.info(f"Original image dimensions: {original_width}x{original_height}")
            
            # Check memory requirements for original image
            memory_check = self._check_image_memory_requirements(original_height, original_width)
            logger.info(f"Memory analysis: {memory_check}")
            
            if not memory_check["can_process"]:
                logger.warning(f"Image may cause memory issues (needs {memory_check['estimated_memory_mb']:.1f}MB, "
                             f"available {memory_check['available_memory_mb']:.1f}MB)")
                # Force garbage collection before processing
                gc.collect()
                if self.device != 'cpu':
                    torch.cuda.empty_cache()
            
            # CRITICAL: Process image at original size to maintain exact coordinates
            frame = original_frame.copy()  # Work with original dimensions
            
            logger.info(f"Processing at original size: {frame.shape}")
            
            # Run detection with explicit device parameter for Railway
            logger.info(f"Running YOLO Large detection on device: {self.device} (ORIGINAL SIZE)")
            with torch.no_grad():  # Disable gradients for inference
                # Process in smaller batches if image is very large
                if memory_check["memory_ratio"] > 0.5:
                    logger.info("Large image detected - using optimized inference settings")
                    results = self.model(
                        frame, 
                        conf=conf_threshold, 
                        verbose=False, 
                        save=False,
                        device=self.device,
                        half=True if self.device != 'cpu' else False,
                        max_det=300,  # Limit max detections for memory
                        agnostic_nms=True  # Faster NMS
                    )
                else:
                    results = self.model(
                        frame, 
                        conf=conf_threshold, 
                        verbose=False, 
                        save=False,
                        device=self.device,
                        half=True if self.device != 'cpu' else False
                    )
            
            if not results:
                logger.warning("No results returned from model")
                return {"error": "No results from detection model", "detections": []}
            
            logger.info("Processing detection results with original coordinates...")
            all_detections = self.process_detections(results, original_frame.shape)
            logger.info(f"Total detections found: {len(all_detections)}")
            
            # Log all detected classes for debugging
            if all_detections:
                detected_classes = [d.class_name for d in all_detections]
                logger.info(f"Detected classes: {set(detected_classes)}")
            
            # Filter to only requested objects
            filtered_detections = self.filter_detections_by_class(all_detections, target_classes)
            logger.info(f"Filtered detections: {len(filtered_detections)}")
            
            # FALLBACK: If no matches found but other objects exist, show all objects
            used_fallback = False
            if len(filtered_detections) == 0 and len(all_detections) > 0 and fallback_to_all:
                filtered_detections = all_detections
                used_fallback = True
                logger.info("Using fallback - showing all detected objects")
            
            # Prepare response with ORIGINAL coordinates
            response = {
                "requested_objects": target_objects,
                "searched_classes": target_classes,
                "total_objects_found": len(all_detections),
                "matching_objects_found": len(filtered_detections),
                "used_fallback": used_fallback,
                "fallback_message": "No requested objects found. Showing all detected objects." if used_fallback else None,
                "image_dimensions": {"width": original_width, "height": original_height},
                "coordinates_type": "ORIGINAL_EXACT",  # Flag indicating coordinates are exact
                "model_type": "YOLO11 Large",
                "device_used": self.device,
                "memory_info": memory_check,
                "detections": []
            }
            
            # Add detections with ORIGINAL coordinates
            for i, detection in enumerate(filtered_detections):
                # These coordinates are exact to the original image
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
                    "vertices": vertices,  # All four corner points - EXACT to original
                    "center_point": {
                        "x": detection.center_point[0],
                        "y": detection.center_point[1]
                    },
                    "area": detection.area,
                    "dimensions": {
                        "width": x2 - x1,
                        "height": y2 - y1
                    },
                    "coordinates_note": "Exact pixel coordinates from original image"
                })
            
            # Add annotated image if requested and detections exist
            if return_annotated_image and len(filtered_detections) > 0:
                try:
                    logger.info("Creating annotated image with ORIGINAL coordinates...")
                    # Draw detections on ORIGINAL image
                    annotated = original_frame.copy()
                    for detection in filtered_detections:
                        x1, y1, x2, y2 = detection.bbox
                        # Different color for fallback detections
                        color = (0, 165, 255) if used_fallback else (0, 255, 0)  # Orange for fallback, green for matches
                        # Draw bounding box with coordinates exact to original
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)  # Thicker lines for Large model
                        # Draw label with better visibility
                        label = f"{detection.class_name}: {detection.confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                        cv2.putText(annotated, label, (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Add fallback message on image if used
                    if used_fallback:
                        cv2.putText(annotated, "Showing all objects (no matches found)", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                    # Add model info with device and coordinate type
                    cv2.putText(annotated, f"YOLO11 Large ({self.device.upper()}) - EXACT COORDS", 
                               (10, annotated.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Convert to base64 for web transmission with higher quality
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Higher quality for Large model
                    _, buffer = cv2.imencode('.jpg', annotated, encode_param)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    response["annotated_image_base64"] = img_base64
                    
                    logger.info(f"Annotated image created with exact coordinates, base64 length: {len(img_base64)}")
                    
                except Exception as e:
                    logger.error(f"Error creating annotated image: {e}")
                    response["annotation_error"] = str(e)
            
            # Memory cleanup and logging
            memory_after = self.get_memory_usage()
            logger.info(f"Memory after detection: {memory_after}")
            
            # Force cleanup for large images
            del original_frame, frame
            if 'annotated' in locals():
                del annotated
            gc.collect()
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in detect_specific_objects: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e), "detections": []}
    
    def detect_from_base64(self, image_base64: str, target_objects: List[str],
                          conf_threshold: float = 0.25, include_similar: bool = True,
                          fallback_to_all: bool = True) -> dict:
        """Detect objects from base64 encoded image (for web API) - PRESERVES EXACT COORDINATES"""
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Failed to decode base64 image", "detections": []}
            
            logger.info(f"Processing base64 image at original size: {img.shape}")
            
            # Save temporarily WITHOUT any modifications to preserve exact dimensions
            temp_path = f"temp_original_{int(time.time())}.jpg"
            # Use maximum quality to preserve image exactly
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            # Run detection
            result = self.detect_specific_objects(
                temp_path, target_objects, conf_threshold, include_similar, 
                return_annotated_image=True, fallback_to_all=fallback_to_all
            )
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detect_from_base64: {e}")
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