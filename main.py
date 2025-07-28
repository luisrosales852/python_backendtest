from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import uuid
from typing import Optional
import logging

from detector import YOLODetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Object Detection API",
    description="YOLO-based object detection service",
    version="1.0.0"
)

# Get environment variables
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11l.pt")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (will be done on startup)
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the YOLO detector on startup"""
    global detector
    try:
        logger.info(f"Initializing YOLO detector with model: {MODEL_PATH}")
        detector = YOLODetector(model_path=MODEL_PATH)
        logger.info("YOLO detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO detector: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_loaded": detector is not None,
        "message": "Object Detection API is running",
        "model_path": MODEL_PATH,
        "cors_origins": CORS_ORIGINS
    }

@app.get("/available_classes")
async def get_available_classes():
    """Get available object classes and categories"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        return {
            "classes": detector.get_available_classes(),
            "categories": detector.get_similar_object_mappings(),
            "total_classes": len(detector.get_available_classes())
        }
    except Exception as e:
        logger.error(f"Error getting available classes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available classes: {str(e)}")

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
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
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
        
        logger.info(f"Processing image: {file.filename}, objects: {target_objects}")
        
        # Run detection
        result = detector.detect_specific_objects(
            temp_path, 
            target_objects, 
            confidence, 
            include_similar, 
            return_annotated_image=True, 
            fallback_to_all=fallback_to_all
        )
        
        # Check for errors in detection
        if "error" in result:
            logger.error(f"Detection error: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {result['error']}")
        
        logger.info(f"Detection completed: {result['matching_objects_found']} objects found")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT) 