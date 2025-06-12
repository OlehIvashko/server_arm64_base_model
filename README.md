# 🎨 Background Removal API

FastAPI-based web service for automatic background removal from images using AI. Built with InSPyReNet model and ONNX runtime for high performance inference.

## ✨ Features

- **🚀 Fast AI-powered background removal** using InSPyReNet model
- **🔄 Two processing modes:**
  - **Mask mode** (fast) - returns alpha mask, composition in browser
  - **Image mode** (convenient) - returns ready-to-use PNG with transparent background
- **📱 EXIF orientation handling** - automatically fixes rotated mobile photos
- **🌐 Base64 API** - simple JSON interface, no file uploads
- **📊 Detailed performance metrics** - processing time breakdown
- **🎯 Beautiful web interface** - drag & drop image upload

## 🏗️ Architecture

- **Backend:** FastAPI + ONNX Runtime + OpenCV
- **Model:** InSPyReNet (inspyrenet_s-coco) converted to ONNX
- **Frontend:** Pure HTML5/CSS3/JavaScript with Canvas API
- **Processing:** 1024×1024 input resolution, optimized for ARM64

## 📋 Requirements

- **Python 3.10+** (tested on 3.11)
- **macOS ARM64** (Apple Silicon) or **Linux ARM64**
- **8GB RAM** minimum (model loading + inference)
- **~1GB disk space** (for model and dependencies)

## 🚀 Quick Start

### Option 1: Local Installation (Recommended)

1. **Clone the repository:**
```bash
git clone <repository-url>
cd server_arm64_base_model
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install PyTorch for CPU:**
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

4. **Install build dependencies:**
```bash
pip install -r requirements-builder.txt
```

5. **Download and convert model:**
```bash
mkdir -p models
curl -L "https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base.pth" -o "models/ckpt_base.pth"
python convert_to_onnx.py
```

6. **Install runtime dependencies:**
```bash
pip install -r requirements-runtime.txt
```

7. **Start the server:**
```bash
python server_app.py
```

8. **Open web interface:**
   - Open `index.html` in your browser
   - Or visit API docs at http://localhost:8000/docs

### Option 2: Docker (Alternative)

```bash
docker build -t background-removal-api .
docker run -p 8000:8000 background-removal-api
```

## 📖 Usage

### Web Interface

1. **Open `index.html`** in your browser
2. **Choose processing mode:**
   - 🎭 **Mask mode** - Fast processing, composition in browser
   - 🖼️ **Image mode** - Slower, returns ready PNG file
3. **Upload image** - drag & drop or click to select
4. **Process** - click "Process Image" button
5. **Download result** - click download button for PNG file

### API Usage

#### Endpoint: `POST /remove-background/`

**Request:**
```json
{
  "image_base64": "base64_encoded_image_data",
  "filename": "image.jpg",
  "mode": "mask"  // "mask" or "image"
}
```

**Response:**
```json
{
  "mask_base64_png": "base64_encoded_mask", // if mode="mask"
  "image_base64_png": "base64_encoded_image", // if mode="image" 
  "mode": "mask",
  "filename": "image.jpg",
  "diagnostics": {
    "total_duration_ms": 6200,
    "preprocess_duration_ms": 150,
    "inference_duration_ms": 5800,
    "postprocess_duration_ms": 250,
    "original_size_wh": [1920, 1080],
    "model_input_size_wh": [1024, 1024],
    "orientation_was_fixed": true
  }
}
```

#### cURL Example:
```bash
# Convert image to base64 first
IMAGE_BASE64=$(base64 -i your_image.jpg)

# Make API request
curl -X POST "http://localhost:8000/remove-background/" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$IMAGE_BASE64\",
    \"filename\": \"your_image.jpg\",
    \"mode\": \"mask\"
  }"
```

#### Python Example:
```python
import base64
import requests
import json

# Read and encode image
with open("your_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/remove-background/",
    json={
        "image_base64": image_base64,
        "filename": "your_image.jpg",
        "mode": "mask"
    }
)

result = response.json()
print(f"Processing took {result['diagnostics']['total_duration_ms']}ms")
```

## ⚡ Performance

**Typical processing times on Apple M1:**
- **Preprocessing:** 100-200ms (resize, normalization)
- **AI Inference:** 5000-6000ms (model execution) 
- **Postprocessing:** 250-300ms (mask mode) / 800-1200ms (image mode)
- **Total:** ~6-7 seconds per image

**Speed comparison:**
- **Mask mode:** 17x faster postprocessing
- **Base64 vs multipart:** No significant difference
- **EXIF fixing:** +50-100ms when needed

## 🔧 Configuration

### Model Settings
```python
MODEL_INPUT_SIZE = (1024, 1024)  # Fixed model input size
MODEL_PATH = "models/ckpt_base.onnx"
```

### Server Settings
```python
# In server_app.py
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### CORS Settings
CORS is enabled for all origins in development. For production, update:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## 📁 Project Structure

```
server_arm64_base_model/
├── server_app.py              # Main FastAPI application
├── convert_to_onnx.py          # Model conversion script
├── index.html                  # Web interface
├── requirements-runtime.txt    # Runtime dependencies
├── requirements-builder.txt    # Build dependencies  
├── Dockerfile                  # Docker configuration
├── models/                     # Model files directory
│   ├── ckpt_base.pth          # Original PyTorch model
│   └── ckpt_base.onnx         # Converted ONNX model
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🐛 Troubleshooting

### Common Issues

**1. "ONNX model not found" error:**
```bash
# Make sure model is downloaded and converted
mkdir -p models
curl -L "https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base.pth" -o "models/ckpt_base.pth"
python convert_to_onnx.py
```

**2. "Invalid input data type" error:**
```bash
# Install compatible NumPy version
pip install "numpy<2.0"
```

**3. "CORS policy" error in browser:**
- Make sure server is running on localhost:8000
- Check browser console for specific error details

**4. Slow processing:**
- Use mask mode for faster processing
- Check available RAM (model needs ~2GB)
- Monitor CPU usage during inference

### Performance Optimization

**For faster processing:**
1. Use **mask mode** instead of image mode
2. Resize large images before processing
3. Use **SSD storage** for model files
4. Ensure adequate **RAM** (8GB+)

**For production deployment:**
1. Use **GPU inference** if available
2. Implement **request queuing** for high load
3. Add **image caching** for repeated requests
4. Set up **load balancing** for multiple instances

## 🔮 Roadmap

- [ ] **GPU acceleration** with CUDA/MPS providers
- [ ] **Batch processing** for multiple images
- [ ] **WebSocket** support for real-time processing
- [ ] **Multiple model formats** (different quality/speed tradeoffs)
- [ ] **Background replacement** instead of just removal
- [ ] **REST API authentication** for production use

## 📄 License

This project uses the following components:
- **InSPyReNet model:** [MIT License](https://github.com/plemeri/transparent-background)
- **FastAPI:** [MIT License](https://github.com/tiangolo/fastapi)
- **ONNX Runtime:** [MIT License](https://github.com/microsoft/onnxruntime)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

If you encounter issues:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review server logs for error details
3. Open an issue with full error messages and system info
4. Include sample images that cause problems (if applicable)

---

**Built with ❤️ for efficient AI-powered background removal** 