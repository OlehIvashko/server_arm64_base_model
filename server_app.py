import base64
import io
import os
import time
from typing import Tuple, Dict, Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ckpt_base.onnx")
MODEL_INPUT_SIZE = (1024, 1024)  # (width, height)

SESSION = None

# --- Pydantic Models ---
class ImageRequest(BaseModel):
    image_base64: str
    filename: str = "image.jpg"
    mode: str = "mask"  # "mask" or "image"

# --- FastAPI App Initialization ---
app = FastAPI(title="Background Removal API", version="1.0.0")

# Add CORS middleware for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def _init_session():
    global SESSION
    if SESSION is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"ONNX model not found at {MODEL_PATH}. Ensure it's copied correctly in Dockerfile.")
        print(f"Loading ONNX model from: {MODEL_PATH}")
        providers = ['CPUExecutionProvider']
        SESSION = ort.InferenceSession(MODEL_PATH, providers=providers)
        print("ONNX model loaded successfully.")
    return SESSION

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

def _preprocess(img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
    original_size_wh = img.size
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
    img_np = np.array(img_resized).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    img_np = (img_np - mean) / std
    img_transposed = img_np.transpose(2, 0, 1)
    return np.expand_dims(img_transposed, axis=0).astype(np.float32), original_size_wh



# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Loads model on server startup."""
    _init_session()

@app.post("/remove-background/", response_class=JSONResponse)
async def remove_background(request: ImageRequest) -> Dict[str, Any]:
    start_total = time.perf_counter()
    try:
        if SESSION is None: # Double check, although startup_event should have triggered
            _init_session()

        # Decode base64
        try:
            image_data = base64.b64decode(request.image_base64)
            original_pil_img = Image.open(io.BytesIO(image_data))
            
            # Fix orientation based on EXIF data
            original_pil_img_fixed, orientation_was_fixed = _fix_image_orientation(original_pil_img)
            original_pil_img_rgb = original_pil_img_fixed.convert("RGB")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

        # Preprocessing
        start_preprocess = time.perf_counter()
        preprocessed_img_tensor, original_size_wh = _preprocess(original_pil_img_rgb)
        preprocess_time = round((time.perf_counter() - start_preprocess) * 1000)

        # Inference (background removal)
        start_inference = time.perf_counter()
        input_name = SESSION.get_inputs()[0].name
        output_name = SESSION.get_outputs()[0].name
        result_onnx = SESSION.run([output_name], {input_name: preprocessed_img_tensor})
        inference_time = round((time.perf_counter() - start_inference) * 1000)

        # Postprocessing - depending on mode
        start_postprocess = time.perf_counter()
        
        # Mask processing
        if result_onnx[0].ndim == 4 and result_onnx[0].shape[1] == 1:
             mask_squeezed = result_onnx[0].squeeze(axis=(0, 1))
        elif result_onnx[0].ndim == 3:
            mask_squeezed = result_onnx[0].squeeze(axis=0)
        else:
            mask_squeezed = result_onnx[0]
            
        # Fast normalization
        mask_normalized = np.clip(mask_squeezed, 0, 1)
        
        # Fast resize with cv2
        import cv2
        mask_resized = cv2.resize(mask_normalized, original_size_wh, interpolation=cv2.INTER_LINEAR)
        alpha_mask = (mask_resized * 255).astype(np.uint8)
        
        if request.mode == "mask":
            # Return only mask
            mask_img = Image.fromarray(alpha_mask, mode='L')
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            result_base64 = base64.b64encode(buf.getvalue()).decode()
            result_key = "mask_base64_png"
        else:
            # Create full image with transparent background
            original_array = np.array(original_pil_img_rgb)
            rgba_array = np.dstack([original_array, alpha_mask])
            final_img = Image.fromarray(rgba_array, 'RGBA')
            
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            result_base64 = base64.b64encode(buf.getvalue()).decode()
            result_key = "image_base64_png"
        
        postprocess_time = round((time.perf_counter() - start_postprocess) * 1000)

        total_time = round((time.perf_counter() - start_total) * 1000)

        print(f"Request processed [{request.mode} mode]. Total: {total_time}ms | Preprocess: {preprocess_time}ms | Inference: {inference_time}ms | Postprocess: {postprocess_time}ms")

        response = {
            result_key: result_base64,
            "mode": request.mode,
            "filename": request.filename,
            "diagnostics": {
                "total_duration_ms": total_time,
                "preprocess_duration_ms": preprocess_time,
                "inference_duration_ms": inference_time,
                "postprocess_duration_ms": postprocess_time,
                "original_size_wh": original_size_wh,
                "model_input_size_wh": MODEL_INPUT_SIZE,
                "orientation_was_fixed": orientation_was_fixed
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
    return {"message": "Background Removal API is running. Use POST /remove-background/ to process an image."}

# For local development (not used by Docker CMD, but useful for development)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)