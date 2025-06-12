import base64
import io
import os
import time
from typing import Tuple, Dict

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ckpt_base.onnx")
MODEL_INPUT_SIZE = (1024, 1024) # (ширина, висота)

SESSION = None

# --- FastAPI App Initialization ---
app = FastAPI(title="Background Removal API", version="1.0.0")

# --- Helper Functions (аналогічні до lambda_handler) ---

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

def _preprocess(img: Image.Image) -> Tuple]:
    original_size_wh = img.size
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
    img_np = np.array(img_resized).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    img_np = (img_np - mean) / std
    img_transposed = img_np.transpose(2, 0, 1)
    return np.expand_dims(img_transposed, axis=0), original_size_wh

def _postprocess(mask_np: np.ndarray, original_size_wh: Tuple[int, int], original_img_rgb: Image.Image) -> Image.Image:
    if mask_np.ndim == 4 and mask_np.shape[1] == 1:
         mask_squeezed = mask_np.squeeze(axis=(0, 1))
    elif mask_np.ndim == 3:
        mask_squeezed = mask_np.squeeze(axis=0)
    elif mask_np.ndim == 2:
        mask_squeezed = mask_np
    else:
        raise ValueError(f"Unexpected mask shape: {mask_np.shape}")

    min_val = np.min(mask_squeezed)
    max_val = np.max(mask_squeezed)
    if max_val - min_val > 1e-6:
        saliency_map_normalized = (mask_squeezed - min_val) / (max_val - min_val)
    else:
        saliency_map_normalized = np.zeros_like(mask_squeezed)

    alpha_mask_pil = Image.fromarray((saliency_map_normalized * 255).astype(np.uint8), mode='L')
    alpha_mask_resized_pil = alpha_mask_pil.resize(original_size_wh, Image.Resampling.LANCZOS)

    final_img_rgba = original_img_rgb.convert("RGBA")
    final_img_rgba.putalpha(alpha_mask_resized_pil)
    return final_img_rgba

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Завантажує модель при старті сервера."""
    _init_session()

@app.post("/remove-background/", response_class=JSONResponse)
async def remove_background(image: UploadFile = File(...)) -> Dict[str, any]:
    start_total = time.perf_counter()
    try:
        if SESSION is None: # Подвійна перевірка, хоча startup_event має спрацювати
            _init_session()

        contents = await image.read()
        original_pil_img_rgb = Image.open(io.BytesIO(contents)).convert("RGB")

        preprocessed_img_tensor, original_size_wh = _preprocess(original_pil_img_rgb)

        start_inference = time.perf_counter()
        input_name = SESSION.get_inputs().name
        output_name = SESSION.get_outputs().name
        result_onnx = SESSION.run([output_name], {input_name: preprocessed_img_tensor})
        inference_time = round((time.perf_counter() - start_inference) * 1000)

        final_pil_img_transparent = _postprocess(result_onnx, original_size_wh, original_pil_img_rgb)

        buf = io.BytesIO()
        final_pil_img_transparent.save(buf, format="PNG")
        image_base64_png = base64.b64encode(buf.getvalue()).decode()

        total_time = round((time.perf_counter() - start_total) * 1000)

        print(f"Request processed. Total time: {total_time}ms, Inference time: {inference_time}ms")

        return {
            "image_base64_png": image_base64_png,
            "filename": image.filename,
            "content_type": image.content_type,
            "diagnostics": {
                "total_duration_ms": total_time,
                "inference_duration_ms": inference_time,
                "original_size_wh": original_size_wh,
                "model_input_size_wh": MODEL_INPUT_SIZE
            }
        }

    except Exception as e:
        import traceback
        error_message = f"Error processing request: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail={"error": error_message, "trace": traceback.format_exc()})

@app.get("/")
async def root():
    return {"message": "Background Removal API is running. Use POST /remove-background/ to process an image."}

# Для локального запуску (не використовується Docker CMD, але корисно для розробки)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)