from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import uuid
from typing import Optional
import logging
import traceback
import psutil
import signal
import sys

from detector import YOLODetector

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Object Detection API - YOLO Large (Exact Coordinates)",
    description="Advanced YOLO11 Large model object detection service with EXACT pixel coordinates - NO image resizing",
    version="2.1.0"
)

# Railway-optimized environment variables
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")  # Railway allows flexible CORS
if CORS_ORIGINS != "*":
    CORS_ORIGINS = CORS_ORIGINS.split(",")
else:
    CORS_ORIGINS = ["*"]

MODEL_PATH = os.getenv("MODEL_PATH", "yolo11l.pt")  # Large model default
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "80"))

# Railway-specific settings for EXACT coordinate detection
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))  # Single worker for memory efficiency
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "unknown")
# INCREASED limits to support large images with exact coordinates
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "104857600"))  # 100MB for exact coordinate detection
MAX_RESPONSE_SIZE = int(os.getenv("MAX_RESPONSE_SIZE", "209715200"))  # 200MB for large annotated images

# Enable CORS for Railway deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize detector (will be done on startup)
detector = None
detector_error = None

def graceful_shutdown(signum, frame):
    """Graceful shutdown handler for Railway"""
    logger.info("Received shutdown signal, cleaning up...")
    global detector
    if detector:
        # Clean up any resources
        pass
    sys.exit(0)

# Register signal handlers for Railway
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

@app.on_event("startup")
async def startup_event():
    """Initialize the YOLO Large detector on startup"""
    global detector, detector_error
    try:
        logger.info("=" * 60)
        logger.info("Starting Object Detection API with YOLO Large")
        logger.info("EXACT COORDINATES MODE - No image resizing")
        logger.info(f"Railway Environment: {RAILWAY_ENVIRONMENT}")
        logger.info("=" * 60)
        
        # Check system resources
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        logger.info(f"System Info:")
        logger.info(f"  - Total Memory: {memory.total / (1024**3):.2f} GB")
        logger.info(f"  - Available Memory: {memory.available / (1024**3):.2f} GB")
        logger.info(f"  - CPU Cores: {cpu_count}")
        logger.info(f"  - Max Image Size: {MAX_IMAGE_SIZE / (1024*1024):.1f} MB")
        
        logger.info(f"Initializing YOLO Large detector with model: {MODEL_PATH}")
        detector = YOLODetector(model_path=MODEL_PATH)
        logger.info("YOLO Large detector initialized successfully")
        
        # Test the detector with a simple call
        available_classes = detector.get_available_classes()
        logger.info(f"Detector test successful. Available classes: {len(available_classes)}")
        
        # Log memory usage after initialization
        memory_after = psutil.virtual_memory()
        logger.info(f"Memory after initialization: {memory_after.available / (1024**3):.2f} GB available")
        
    except Exception as e:
        error_msg = f"Failed to initialize YOLO Large detector: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        detector_error = error_msg
        # Don't raise - let the app start but return errors in endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint with system information"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        "status": "healthy" if detector is not None else "detector_error",
        "detector_loaded": detector is not None,
        "detector_error": detector_error,
        "message": "Object Detection API with YOLO Large - EXACT COORDINATES",
        "model_path": MODEL_PATH,
        "model_type": "YOLO11 Large",
        "coordinate_mode": "EXACT_ORIGINAL",
        "max_image_size_mb": MAX_IMAGE_SIZE / (1024 * 1024),
        "cors_origins": CORS_ORIGINS,
        "environment": RAILWAY_ENVIRONMENT,
        "system": {
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_used_percent": memory.percent,
            "cpu_percent": cpu_percent,
            "cpu_count": psutil.cpu_count()
        }
    }

@app.get("/available_classes")
async def get_available_classes():
    """Get available object classes and categories"""
    if detector is None:
        error_msg = f"Detector not initialized. Error: {detector_error}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    try:
        # Try to get classes with error handling
        classes = detector.get_available_classes()
        categories = detector.get_similar_object_mappings()
        
        return {
            "classes": classes,
            "categories": categories,
            "total_classes": len(classes),
            "model_type": "YOLO11 Large",
            "coordinate_mode": "EXACT_ORIGINAL"
        }
    except Exception as e:
        logger.error(f"Error getting available classes: {e}")
        logger.error(traceback.format_exc())
        
        # Fallback to hardcoded YOLO classes if model fails
        fallback_classes = [
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
        
        fallback_categories = {
            "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "person": ["person"],
            "animal": ["cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "food": ["banana", "apple", "sandwich", "orange", "pizza", "donut", "cake"]
        }
        
        return {
            "classes": fallback_classes,
            "categories": fallback_categories,
            "total_classes": len(fallback_classes),
            "fallback_used": True,
            "model_type": "YOLO11 Large (Fallback)",
            "coordinate_mode": "EXACT_ORIGINAL",
            "error": str(e)
        }

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    objects: str = Form(...),  # Comma-separated list
    include_similar: bool = Form(True),
    confidence: float = Form(0.25),  # Lower default for Large model
    fallback_to_all: bool = Form(True)
):
    """
    Detect objects in uploaded image using YOLO Large with EXACT pixel coordinates
    
    - **file**: Image file to analyze (max 100MB for exact coordinate detection)
    - **objects**: Comma-separated list of objects to detect
    - **include_similar**: Whether to include similar objects (default: True)
    - **confidence**: Confidence threshold 0-1 (default: 0.25 for Large model)
    - **fallback_to_all**: Show all detected objects if no matches found (default: True)
    
    **IMPORTANT**: All coordinates returned are EXACT pixel locations from your original image.
    No resizing is performed, ensuring precise bounding box coordinates.
    """
    if detector is None:
        error_msg = f"Detector not initialized. Error: {detector_error}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (increased limit for exact coordinate detection)
    if file.size and file.size > MAX_IMAGE_SIZE:
        size_mb = MAX_IMAGE_SIZE / (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File size must be less than {size_mb}MB for exact coordinate detection")
    
    # Validate confidence range
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
    
    # Parse target objects
    target_objects = [obj.strip() for obj in objects.split(",") if obj.strip()]
    if not target_objects:
        raise HTTPException(status_code=400, detail="At least one object must be specified")
    
    temp_path = None
    try:
        # Log memory before processing
        memory_before = psutil.virtual_memory()
        logger.info(f"Memory before processing: {memory_before.available / (1024**3):.2f} GB available")
        
        # Create temporary file with unique name
        temp_dir = os.getenv("TEMP_DIR", "/tmp")  # Railway uses /tmp
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"detect_exact_{uuid.uuid4().hex}_{file.filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save uploaded file with maximum quality to preserve exact dimensions
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        logger.info(f"Processing image for EXACT coordinates: {file.filename}")
        logger.info(f"  - Objects: {target_objects}")
        logger.info(f"  - Confidence: {confidence}")
        logger.info(f"  - File size: {file_size_mb:.2f} MB")
        logger.info(f"  - Temp file: {temp_path}")
        
        # Run detection with error catching - PRESERVES ORIGINAL DIMENSIONS
        try:
            result = detector.detect_specific_objects(
                temp_path, 
                target_objects, 
                confidence, 
                include_similar, 
                return_annotated_image=True, 
                fallback_to_all=fallback_to_all
            )
        except Exception as detection_error:
            logger.error(f"Detection processing error: {detection_error}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Detection processing failed: {str(detection_error)}")
        
        # Check for errors in detection
        if "error" in result:
            logger.error(f"Detection error: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {result['error']}")
        
        # Log detailed results
        logger.info(f"Detection completed successfully (EXACT COORDINATES):")
        logger.info(f"  - Total objects found: {result.get('total_objects_found', 0)}")
        logger.info(f"  - Matching objects: {result.get('matching_objects_found', 0)}")
        logger.info(f"  - Used fallback: {result.get('used_fallback', False)}")
        logger.info(f"  - Original image dimensions: {result.get('image_dimensions', {})}")
        logger.info(f"  - Coordinate type: {result.get('coordinates_type', 'Unknown')}")
        logger.info(f"  - Model type: {result.get('model_type', 'Unknown')}")
        logger.info(f"  - Device used: {result.get('device_used', 'Unknown')}")
        
        # Enhanced response size management for large exact-coordinate images
        # Estimate response size (rough approximation)
        if "annotated_image_base64" in result and result["annotated_image_base64"]:
            estimated_size = len(result["annotated_image_base64"]) * 0.75  # Base64 is ~33% larger than binary
            logger.info(f"Estimated response size: {estimated_size / (1024*1024):.2f} MB")
            
            if estimated_size > MAX_RESPONSE_SIZE:
                logger.warning(f"Response too large ({estimated_size / (1024*1024):.2f} MB), removing annotated image")
                result["annotated_image_base64"] = None
                result["image_too_large"] = True
                result["message"] = "Annotated image removed due to size. Detection coordinates preserved and exact."
        
        # Log memory after processing
        memory_after = psutil.virtual_memory()
        logger.info(f"Memory after processing: {memory_after.available / (1024**3):.2f} GB available")
        
        # Add response metadata for exact coordinates
        result["api_version"] = "2.1.0"
        result["processing_mode"] = "EXACT_COORDINATES"
        result["image_resized"] = False
        result["coordinate_accuracy"] = "PIXEL_PERFECT"
        
        return JSONResponse(
            content=result,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "X-Coordinate-Mode": "EXACT_ORIGINAL",
                "X-Image-Resized": "false"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during exact coordinate detection: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")

# Add OPTIONS handler for CORS preflight
@app.options("/detect")
async def detect_options():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/stats")
async def get_system_stats():
    """Get current system statistics - useful for Railway monitoring"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    stats = {
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        },
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count()
        },
        "model_type": "YOLO11 Large",
        "coordinate_mode": "EXACT_ORIGINAL",
        "max_image_size_mb": MAX_IMAGE_SIZE / (1024 * 1024),
        "max_response_size_mb": MAX_RESPONSE_SIZE / (1024 * 1024),
        "detector_loaded": detector is not None
    }
    
    if detector:
        try:
            detector_memory = detector.get_memory_usage()
            stats["detector_memory"] = detector_memory
        except:
            pass
    
    return stats

@app.get("/coordinate_info")
async def get_coordinate_info():
    """Get information about coordinate accuracy and processing mode"""
    return {
        "coordinate_mode": "EXACT_ORIGINAL",
        "image_resizing": False,
        "coordinate_accuracy": "PIXEL_PERFECT",
        "description": "All bounding box coordinates are exact pixel locations from your original image. No resizing is performed.",
        "supported_formats": ["JPG", "JPEG", "PNG"],
        "max_image_size_mb": MAX_IMAGE_SIZE / (1024 * 1024),
        "benefits": [
            "Exact pixel coordinates",
            "No coordinate scaling needed",
            "Perfect for pixel-level analysis",
            "Maintains original image quality",
            "Suitable for high-precision applications"
        ],
        "note": "Large images may require more processing time and memory but coordinates remain exact."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT,
        workers=MAX_WORKERS,
        access_log=True,
        log_level="info"
    ) 