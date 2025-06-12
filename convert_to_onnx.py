# convert_to_onnx.py
import torch
import os
import sys
from transparent_background import Remover

def convert():
    """
    Завантажує модель ckpt_base.pth (inspyrenet_s-coco.pth) та конвертує її у формат ONNX.
    """
    print("Starting ONNX conversion for InSPyReNet 'base' model (inspyrenet_s-coco)...")
    # WORKDIR встановлено в /app у Dockerfile, тому шляхи відносні
    model_dir = "models"
    # Модель буде завантажена як ckpt_base.pth
    pth_filename = "ckpt_base.pth"
    onnx_filename = "ckpt_base.onnx" # Вихідний файл ONNX

    pth_path = os.path.join(model_dir, pth_filename)
    onnx_path = os.path.join(model_dir, onnx_filename)

    if not os.path.exists(pth_path):
        print(f"Error: PyTorch model not found at {pth_path}")
        sys.exit(1)

    # Ініціалізуємо Remover з mode='base' для завантаження правильної архітектури моделі.
    # jit=False необхідно для отримання стандартної моделі PyTorch, яку можна експортувати.
    try:
        print(f"Loading model from {pth_path} using transparent_background.Remover (mode='base')...")
        remover = Remover(mode="base", ckpt=pth_path, jit=False, device='cpu')
        model = remover.model
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model with Remover: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

    # Визначаємо розмір вхідних даних для моделі 'base' - 1024x1024.
    # Це значення взято з конфігурації бібліотеки transparent-background для inspyrenet_s-coco.
    dummy_input_size = (1, 3, 1024, 1024)
    dummy_input = torch.randn(dummy_input_size, device='cpu')

    print(f"Exporting model to ONNX with input size {dummy_input_size} to {onnx_path}...")

    try:
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          input_names=['input'],
                          output_names=['output'],
                          opset_version=12,
                          dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}},
                          verbose=False)
        print(f"Model successfully converted to ONNX: {onnx_path}")
    except Exception as e:
        import traceback
        print(f"Error during ONNX conversion: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    convert()