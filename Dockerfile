# Stage 1: Builder - Використовує стандартний ARM Python образ для стабільної крос-компіляції.
FROM --platform=linux/arm64 arm64v8/python:3.10-slim AS builder

WORKDIR /app

# Встановлюємо curl
RUN apt-get update && apt-get install -y curl --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Копіюємо requirements для етапу збірки
COPY requirements-builder.txt.

# Встановлюємо PyTorch для CPU на ARM
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Встановлюємо решту залежностей для збірки
RUN pip install --no-cache-dir -r requirements-builder.txt

# Створюємо директорію для моделей та завантажуємо "base" модель
RUN mkdir -p models
RUN curl -L "https://github.com/plemeri/transparent-background/releases/download/v1.2.12/inspyrenet_s-coco.pth" -o "models/ckpt_base.pth"

# Копіюємо та запускаємо скрипт конвертації
COPY convert_to_onnx.py.
RUN python3.10 convert_to_onnx.py

# ---

# Stage 2: Final - Легкий фінальний образ для сервера.
FROM --platform=linux/arm64 arm64v8/python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Встановлюємо лише рантайм-залежності (включаючи FastAPI, Uvicorn)
COPY requirements-runtime.txt.
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Створюємо директорію для моделей у фінальному образі
RUN mkdir -p models

# Копіюємо лише конвертовану ONNX "base" модель з етапу збірки
COPY --from=builder /app/models/ckpt_base.onnx /app/models/ckpt_base.onnx

# Копіюємо код серверного застосунку
COPY server_app.py.

# Відкриваємо порт, на якому буде працювати FastAPI (за замовчуванням 8000 для Uvicorn)
EXPOSE 8000

# Команда для запуску FastAPI сервера за допомогою Uvicorn
# server_app -- назва файлу server_app.py
# app -- назва екземпляру FastAPI у файлі server_app.py
CMD ["uvicorn", "server_app:app", "--host", "0.0.0.0", "--port", "8000"]