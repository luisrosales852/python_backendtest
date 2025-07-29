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

from detector import YOLODetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Object Detection API",
    description="YOLO-based object detection service",
    version="1.0.0"
)

# Get environment variables with better defaults for production
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")  # Allow all origins in production
if CORS_ORIGINS != "*":
    CORS_ORIGINS = CORS_ORIGINS.split(",")
else:
    CORS_ORIGINS = ["*"]

MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Enable CORS with more permissive settings for production
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

@app.on_event("startup")
async def startup_event():
    """Initialize the YOLO detector on startup"""
    global detector, detector_error
    try:
        logger.info(f"Initializing YOLO detector with model: {MODEL_PATH}")
        detector = YOLODetector(model_path=MODEL_PATH)
        logger.info("YOLO detector initialized successfully")
        
        # Test the detector with a simple call
        available_classes = detector.get_available_classes()
        logger.info(f"Detector test successful. Available classes: {len(available_classes)}")
        
    except Exception as e:
        error_msg = f"Failed to initialize YOLO detector: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        detector_error = error_msg
        # Don't raise - let the app start but return errors in endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if detector is not None else "detector_error",
        "detector_loaded": detector is not None,
        "detector_error": detector_error,
        "message": "Object Detection API is running",
        "model_path": MODEL_PATH,
        "cors_origins": CORS_ORIGINS,
        "environment": os.getenv("RENDER", "local")
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
            "total_classes": len(classes)
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
            "error": str(e)
        }

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    objects: str = Form(...),  # Comma-separated list
    include_similar: bool = Form(True),
    confidence: float = Form(0.3),
    fallback_to_all: bool = Form(True)
):
    """
    Detect objects in uploaded image
    
    - **file**: Image file to analyze
    - **objects**: Comma-separated list of objects to detect
    - **include_similar**: Whether to include similar objects (default: True)
    - **confidence**: Confidence threshold 0-1 (default: 0.3)
    - **fallback_to_all**: Show all detected objects if no matches found (default: True)
    """
    if detector is None:
        error_msg = f"Detector not initialized. Error: {detector_error}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence range
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
    
    # Parse target objects
    target_objects = [obj.strip() for obj in objects.split(",") if obj.strip()]
    if not target_objects:
        raise HTTPException(status_code=400, detail="At least one object must be specified")
    
    temp_path = None
    try:
        # Create temporary file with unique name
        temp_dir = os.getenv("TEMP_DIR", tempfile.gettempdir())
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"detect_{uuid.uuid4().hex}_{file.filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing image: {file.filename}, objects: {target_objects}, confidence: {confidence}")
        logger.info(f"Temp file saved: {temp_path}, size: {os.path.getsize(temp_path)} bytes")
        
        # Run detection with error catching
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
        logger.info(f"Detection completed successfully:")
        logger.info(f"  - Total objects found: {result.get('total_objects_found', 0)}")
        logger.info(f"  - Matching objects: {result.get('matching_objects_found', 0)}")
        logger.info(f"  - Used fallback: {result.get('used_fallback', False)}")
        logger.info(f"  - Image dimensions: {result.get('image_dimensions', {})}")
        
        # For large responses, optionally exclude the base64 image to reduce size
        response_size_limit = int(os.getenv("MAX_RESPONSE_SIZE", "10485760"))  # 10MB default
        
        # Estimate response size (rough approximation)
        if "annotated_image_base64" in result:
            estimated_size = len(result["annotated_image_base64"]) * 0.75  # Base64 is ~33% larger than binary
            logger.info(f"Estimated response size: {estimated_size} bytes")
            
            if estimated_size > response_size_limit:
                logger.warning(f"Response too large ({estimated_size} bytes), removing annotated image")
                result["annotated_image_base64"] = None
                result["image_too_large"] = True
        
        return JSONResponse(
            content=result,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during detection: {e}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT) 