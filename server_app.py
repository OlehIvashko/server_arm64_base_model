import base64
import io
import os
import time
import yaml
from typing import Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transparent_background import Remover
import uvicorn

# --- Configuration ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models", "config.yaml")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Global remover instance
REMOVER = None
CURRENT_MODEL = None

# --- Load model configuration ---
def load_model_config() -> Dict[str, Any]:
    """Load model configuration from config.yaml"""
    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError(f"Model config not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# --- Pydantic Models ---
class ImageRequest(BaseModel):
    image_base64: str
    filename: str = "image.jpg"
    mode: str = "mask"  # "mask" or "image"
    model_type: str = "base"  # "base", "fast", or "base-nightly"

# --- FastAPI App Initialization ---
app = FastAPI(title="Background Removal API", version="2.0.0")

# Add CORS middleware for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def _get_optimal_device():
    """Determine the best available device for inference"""
    import torch
    
    if torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return 'cuda'  # NVIDIA GPU
    else:
        return 'cpu'   # CPU fallback

def _init_remover(model_type: str = "base") -> Remover:
    """Initialize transparent-background Remover with specified model type"""
    global REMOVER, CURRENT_MODEL
    
    if REMOVER is None or CURRENT_MODEL != model_type:
        print(f"Loading {model_type} model...")
        
        # Validate model type
        config = load_model_config()
        if model_type not in config:
            available_models = list(config.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Model type '{model_type}' not available. Available models: {available_models}"
            )
        
        try:
            # # Force CPU usage only
            # device = 'cpu'
            # print(f"Using device: {device} (forced CPU-only mode)")
            device = _get_optimal_device()
            print(f"Using device: {device}")
            # Check if local model file exists
            model_info = config[model_type]
            local_model_path = os.path.join(MODELS_DIR, model_info['ckpt_name'])
            
            if os.path.exists(local_model_path):
                print(f"Using local model: {local_model_path}")
                # Initialize remover with local model path
                # REMOVER = Remover(mode=model_type, ckpt=local_model_path, device='cpu')
                REMOVER = Remover(mode=model_type, ckpt=local_model_path, device=device)
            else:
                print(f"Local model not found, will download automatically...")
                # Initialize remover without path (will auto-download)
                # REMOVER = Remover(mode=model_type, device='cpu')
                REMOVER = Remover(mode=model_type, device=device)
            CURRENT_MODEL = model_type
            
            print(f"Model '{model_type}' loaded successfully on {device}.")
            print(f"Base size: {model_info['base_size']}")
            
        except Exception as e:
            print(f"Error loading model '{model_type}': {e}")
            # Fallback to CPU if GPU fails
            if device != 'cpu':
                print("Falling back to CPU...")
                try:
                    if os.path.exists(local_model_path):
                        REMOVER = Remover(mode=model_type, ckpt=local_model_path, device='cpu')
                    else:
                        REMOVER = Remover(mode=model_type, device='cpu')
                    CURRENT_MODEL = model_type
                    print(f"Model '{model_type}' loaded successfully on CPU.")
                except Exception as cpu_error:
                    print(f"CPU fallback also failed: {cpu_error}")
                    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return REMOVER

def _fix_image_orientation(img: Image.Image) -> Tuple[Image.Image, bool]:
    """Fixes image orientation based on EXIF data"""
    try:
        original_size = img.size
        # Use ImageOps.exif_transpose for automatic orientation correction
        img_fixed = ImageOps.exif_transpose(img)
        if img_fixed is not None:
            was_rotated = img_fixed.size != original_size
            if was_rotated:
                print(f"EXIF orientation fixed: {original_size} -> {img_fixed.size}")
            return img_fixed, was_rotated
        else:
            return img, False
    except Exception as e:
        print(f"Error fixing orientation: {e}")
        # If error occurred, return original image
        return img, False

def _resize_with_padding(img: Image.Image, target_size: Tuple[int, int], pad_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Resize image to target size while preserving aspect ratio using padding
    
    Args:
        img: Input PIL Image
        target_size: Target size as (width, height)
        pad_color: Color for padding (default: black)
    
    Returns:
        Resized image with padding
    """
    target_width, target_height = target_size
    original_width, original_height = img.size
    
    # Calculate scaling factor to fit image within target size
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image maintaining aspect ratio
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and pad color
    padded_img = Image.new("RGB", target_size, pad_color)
    
    # Calculate position to paste resized image (center it)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # Paste resized image onto padded background
    padded_img.paste(resized_img, (paste_x, paste_y))
    
    return padded_img

def _get_model_info(model_type: str) -> Dict[str, Any]:
    """Get model information from config"""
    config = load_model_config()
    if model_type not in config:
        return {}
    return config[model_type]

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Loads default model on server startup."""
    try:
        _init_remover("base")  # Load default model
    except Exception as e:
        print(f"Warning: Could not load default model on startup: {e}")

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    try:
        config = load_model_config()
        models_info = {}
        
        for model_name, model_config in config.items():
            models_info[model_name] = {
                "name": model_name,
                "base_size": model_config.get("base_size", [1024, 1024]),
                "description": f"InspyreNet {model_name} model"
            }
        
        return {
            "available_models": models_info,
            "current_model": CURRENT_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model config: {str(e)}")

@app.post("/remove-background/", response_class=JSONResponse)
async def remove_background(request: ImageRequest) -> Dict[str, Any]:
    start_total = time.perf_counter()
    
    try:
        # Initialize remover with requested model
        remover = _init_remover(request.model_type)
        model_info = _get_model_info(request.model_type)
        
        # Decode base64
        try:
            image_data = base64.b64decode(request.image_base64)
            original_pil_img = Image.open(io.BytesIO(image_data))
            
            # Fix orientation based on EXIF data
            original_pil_img_fixed, orientation_was_fixed = _fix_image_orientation(original_pil_img)
            original_pil_img_rgb = original_pil_img_fixed.convert("RGB")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

        # Preprocessing - prepare image for model
        start_preprocess = time.perf_counter()
        
        # Get original size
        original_size_wh = original_pil_img_rgb.size
        model_base_size = model_info.get("base_size", [512, 512])
        
        # Resize image to model size while preserving aspect ratio
        model_size_tuple = tuple(model_base_size)  # Convert [w, h] to (w, h)
        img_resized_for_model = _resize_with_padding(original_pil_img_rgb, model_size_tuple)
        
        preprocess_time = round((time.perf_counter() - start_preprocess) * 1000)

        # Background removal using transparent-background (inference)
        start_inference = time.perf_counter()
        
        # Process with transparent-background
        result_img = remover.process(img_resized_for_model)
        
        inference_time = round((time.perf_counter() - start_inference) * 1000)

        # Postprocessing - depending on mode
        start_postprocess = time.perf_counter()
        
        if request.mode == "mask":
            # Extract alpha channel as mask
            if result_img.mode == "RGBA":
                # Extract alpha channel
                alpha_channel = result_img.split()[-1]  # Get alpha channel
                mask_img = alpha_channel
                
                # Convert to PNG (keep model size)
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                result_base64 = base64.b64encode(buf.getvalue()).decode()
                result_key = "mask_base64_png"
            else:
                raise HTTPException(status_code=500, detail="Model did not return RGBA image")
        else:
            # Return full RGBA image (keep model size)
            if result_img.mode != "RGBA":
                # Convert to RGBA if not already
                result_img = result_img.convert("RGBA")
            
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            result_base64 = base64.b64encode(buf.getvalue()).decode()
            result_key = "image_base64_png"
        
        postprocess_time = round((time.perf_counter() - start_postprocess) * 1000)
        total_time = round((time.perf_counter() - start_total) * 1000)

        print(f"Request processed [{request.mode} mode, {request.model_type} model]. "
              f"Total: {total_time}ms | Preprocess: {preprocess_time}ms | Inference: {inference_time}ms | Postprocess: {postprocess_time}ms")

        # Get actual result size (should be model size)
        result_size_wh = result_img.size if result_img else model_base_size
        
        # Get device info (forced CPU mode)
        # device_used = 'cpu'
        # Get device info
        device_used = _get_optimal_device()
        response = {
            result_key: result_base64,
            "mode": request.mode,
            "model_type": request.model_type,
            "filename": request.filename,
            "diagnostics": {
                "total_duration_ms": total_time,
                "preprocess_duration_ms": preprocess_time,
                "inference_duration_ms": inference_time,
                "postprocess_duration_ms": postprocess_time,
                "original_size_wh": original_size_wh,
                "model_base_size_wh": model_base_size,
                "result_size_wh": result_size_wh,
                "orientation_was_fixed": orientation_was_fixed,
                "model_used": request.model_type,
                "device_used": device_used
            }
        }
        return response

    except Exception as e:
        import traceback
        error_message = f"Error processing request: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={"error": error_message, "trace": traceback.format_exc()})

@app.get("/")
async def root():
    return {
        "message": "Background Removal API is running. Use POST /remove-background/ to process an image.",
        "version": "2.0.0",
        "models": "Use GET /models to see available models"
    }

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)