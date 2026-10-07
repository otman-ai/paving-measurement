"""Modal deployment definition for the FastAPI pavement-analysis service.

Deploy with: modal deploy modal_app.py
"""

import modal

APP_NAME = "paving-measurement"
GPU_TYPE = "T4"
MODEL_CACHE_PATH = "/cache/huggingface"

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("paving-measurement-hf-cache", create_if_missing=True)
secrets = modal.Secret.from_name(
    "paving-measurement-secrets", required_keys=["HF_TOKEN", "MAPBOX_TOKEN"]
)
google_secrets = modal.Secret.from_name("paving-measurement-google", required_keys=["GOOGLE_MAPS_API_KEY"])

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "fastapi>=0.115",
        "matplotlib>=3.8",
        "numpy>=1.26",
        "opencv-python-headless>=4.10",
        "Pillow>=10.0",
        "python-dotenv>=1.0",
        "requests>=2.31",
        "huggingface_hub>=0.26",
        "ultralytics>=8.3",
        # RF-DETR 1.6+ requires Transformers 5 and a newer torch API than
        # the service's pinned Torch 2.5.1.  1.5.2 supports RFDETRSegNano
        # while remaining compatible with this CUDA/Torch image.
        "rfdetr==1.5.2",
    )
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env({
        "PYTHONPATH": "/root/src",
        "HF_HOME": MODEL_CACHE_PATH,
        "MODEL_ID": "otmanheddouch/yolo26n-seg",
    })
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    secrets=[secrets, google_secrets],
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=3)
@modal.asgi_app(label="api")
def fastapi_app():
    """Expose the FastAPI application on a Modal-managed HTTPS endpoint."""
    from paving_measurement.api import create_api

    return create_api()
