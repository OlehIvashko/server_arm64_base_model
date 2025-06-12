# 🎨 Background Removal API

FastAPI-based web service for automatic background removal from images using AI. Built with transparent-background library and multiple InSPyReNet models for optimal performance.

## ✨ Features

- **🚀 Fast AI-powered background removal** using InSPyReNet models
- **🎯 Multiple AI models:**
  - **Base** (1024×1024) - best quality for high-resolution images
  - **Fast** (384×384) - faster processing for quick results  
  - **Base-nightly** (1024×1024) - experimental features and improvements
- **🔄 Two processing modes:**
  - **Mask mode** (fast) - returns alpha mask, composition in browser
  - **Image mode** (convenient) - returns ready-to-use PNG with transparent background
- **📱 EXIF orientation handling** - automatically fixes rotated mobile photos
- **🌐 Base64 API** - simple JSON interface, no file uploads
- **📊 Detailed performance metrics** - processing time breakdown
- **🎯 Beautiful web interface** - drag & drop image upload with model selection

## 🏗️ Architecture

- **Backend:** FastAPI + transparent-background library
- **Models:** InSPyReNet (base, fast, base-nightly) with automatic downloading
- **Frontend:** Pure HTML5/CSS3/JavaScript with Canvas API
- **Processing:** Variable resolution (384×384 to 1024×1024) depending on model

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

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Or use the startup script:**
```bash
./run_server.sh
```

Note: Models will be automatically downloaded on first use. No manual model setup required!

**Important:** If you encounter package conflicts, use the exact versions from requirements.txt which include compatible flet and albumentations versions.

4. **Start the server:**
```bash
python server_app.py
```

5. **Open web interface:**
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
3. **Choose AI model:**
   - 🎯 **Base** - Best quality (1024×1024)
   - ⚡ **Fast** - Faster processing (384×384)
   - 🌙 **Nightly** - Experimental features (1024×1024)
4. **Upload image** - drag & drop or click to select
5. **Process** - click "Process Image" button
6. **Download result** - click download button for PNG file

### API Usage

#### Endpoint: `POST /remove-background/`

**Request:**
```json
{
  "image_base64": "base64_encoded_image_data",
  "filename": "image.jpg",
  "mode": "mask",  // "mask" or "image"
  "model_type": "base"  // "base", "fast", or "base-nightly"
}
```

**Response:**
```json
{
  "mask_base64_png": "base64_encoded_mask", // if mode="mask"
  "image_base64_png": "base64_encoded_image", // if mode="image" 
  "mode": "mask",
  "model_type": "base",
  "filename": "image.jpg",
  "diagnostics": {
    "total_duration_ms": 6200,
    "inference_duration_ms": 5800,
    "postprocess_duration_ms": 250,
    "original_size_wh": [1920, 1080],
    "model_base_size_wh": [1024, 1024],
    "orientation_was_fixed": true,
    "model_used": "base"
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
     \"mode\": \"mask\",
     \"model_type\": \"base\"
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
        "mode": "mask",
        "model_type": "base"
    }
)

result = response.json()
print(f"Processing took {result['diagnostics']['total_duration_ms']}ms")
```

## ⚡ Performance

**Typical processing times on Apple M1:**
- **Base model (1024×1024):** 4000-6000ms total
- **Fast model (384×384):** 1000-2000ms total  
- **Postprocessing:** 100-300ms (mask mode) / 400-800ms (image mode)

**Speed comparison by model:**
- **Fast model:** ~3x faster than Base model
- **Mask mode:** 2-3x faster postprocessing than image mode
- **EXIF fixing:** +50-100ms when needed

**Model quality vs speed:**
- **Base:** Best quality, slower processing
- **Fast:** Good quality, much faster processing
- **Base-nightly:** Experimental features, similar speed to base

## 🔧 Configuration

### Model Settings
Models are configured in `models/config.yaml`:
```yaml
base:
  url: "https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base.pth"
  base_size: [1024, 1024]

fast:
  url: "https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_fast.pth"
  base_size: [384, 384]
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
├── index.html                  # Web interface
├── requirements-runtime.txt    # Runtime dependencies
├── requirements-builder.txt    # Build dependencies  
├── Dockerfile                  # Docker configuration
├── models/                     # Model files directory
│   └── config.yaml            # Model configuration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

Note: Model files (.pth) are automatically downloaded by transparent-background library on first use.

## 🐛 Troubleshooting

### Common Issues

**1. "Model config not found" error:**
```bash
# Make sure config.yaml exists in models directory
ls models/config.yaml
```

**2. "Model download failed" error:**
- Check internet connection
- Models are downloaded automatically on first use
- Large files (100-400MB) may take time to download

**3. "Invalid input data type" error:**
```bash
# Install compatible NumPy version
pip install "numpy<2.0"
```

**4. "CORS policy" error in browser:**
- Make sure server is running on localhost:8000
- Check browser console for specific error details

**5. Slow processing:**
- Use **fast model** for quicker results
- Use **mask mode** for faster postprocessing
- Check available RAM (models need 1-3GB)
- Monitor CPU usage during inference

### Performance Optimization

**For faster processing:**
1. Use **fast model** (384×384) for quick results
2. Use **mask mode** instead of image mode  
3. Choose appropriate model based on needs:
   - **Fast**: General use, good speed/quality balance
   - **Base**: High quality requirements
4. Ensure adequate **RAM** (8GB+)

**For production deployment:**
1. Use **GPU inference** with transparent-background GPU support
2. Implement **request queuing** for high load
3. Add **image caching** for repeated requests
4. Choose **fast model** for high-volume scenarios
5. Set up **load balancing** for multiple instances

## 🔮 Roadmap

- [ ] **GPU acceleration** with CUDA/Metal support
- [ ] **Batch processing** for multiple images
- [ ] **WebSocket** support for real-time processing
- [ ] **Custom model training** and fine-tuning
- [ ] **Background replacement** instead of just removal
- [ ] **REST API authentication** for production use
- [ ] **Model caching** and optimization
- [ ] **Video processing** support

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