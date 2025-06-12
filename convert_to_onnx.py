# convert_to_onnx.py
import torch
import os
import sys
from transparent_background import Remover

def convert():
    """
    Loads ckpt_base.pth (inspyrenet_s-coco.pth) model and converts it to ONNX format.
    """
    print("Starting ONNX conversion for InSPyReNet 'base' model (inspyrenet_s-coco)...")
    # Model directory for local development
    model_dir = "models"
    # Model will be loaded as ckpt_base.pth
    pth_filename = "ckpt_base.pth"
    onnx_filename = "ckpt_base.onnx"  # Output ONNX file

    pth_path = os.path.join(model_dir, pth_filename)
    onnx_path = os.path.join(model_dir, onnx_filename)

    if not os.path.exists(pth_path):
        print(f"Error: PyTorch model not found at {pth_path}")
        sys.exit(1)

    # Initialize Remover with mode='base' to load the correct model architecture.
    # jit=False is needed to get a standard PyTorch model that can be exported.
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

    # Define input size for 'base' model - 1024x1024.
    # This value is taken from transparent-background library configuration for inspyrenet_s-coco.
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