# Paving Measurement

A pavement-analysis service with a local Gradio interface and a FastAPI deployment target. It retrieves a 3x3 Mapbox satellite-image mosaic, uses SAM 3 to segment pavement, and estimates the covered area. It evaluates multiple grid sizes (1x1 through 7x7) to capture features at different scales.

The reported area is a Web Mercator approximation. It is useful for exploratory analysis and should not be used as a survey-grade measurement.

## Demo

### Before segmentation

![Satellite image before segmentation](assets/before.jpg)

### After segmentation

![Satellite image with asphalt segmentation mask](assets/asphalt-segmentation-result.png)

### Parking lot prompt

The same workflow can use `parking lot` as the segmentation prompt.

#### Before segmentation

![Parking lot satellite image before segmentation](assets/original_parking_lot.webp)

#### After segmentation

![Parking lot segmentation result](assets/after_parking_lot.webp)

## Analysis

The application presents the final unified asphalt mask and a comparison of outputs from each grid size. The grid strategy helps identify pavement at different feature scales.

![Paving measurement analysis view](assets/demo-1.png)

### Auto-labeling potential

This workflow can also accelerate creation of training data for a future customer-specific pavement model. SAM 3 masks can be exported as candidate labels for satellite images, allowing reviewers to correct only inaccurate boundaries instead of annotating every pavement area from scratch. The reviewed masks can then form a higher-quality, customer-specific dataset for training and evaluating a dedicated model.

Use auto-generated masks as a labeling aid, not as ground truth: retain the source imagery, record reviewer corrections, and define consistent labeling rules before model training.

## Project layout

```text
src/paving_measurement/
  api.py            FastAPI endpoints for programmatic analysis
  app.py            Gradio UI and application entry point
  config.py         validated environment-based settings
  geospatial.py     tile coordinates and area calculations
  image_ops.py      image tiling, overlays, and comparisons
  mapbox.py         Mapbox API client
  segmentation.py   SAM 3 inference and multi-grid pipeline
  satellite.py      Mapbox satellite mosaic service
modal_app.py        Modal GPU deployment definition for FastAPI
parking_modal_app.py Modal GPU deployment for tiled parking-stall detection
notebooks/          repeatable experiments
assets/             README images and other lightweight static assets
tests/              focused unit tests
```

## Prerequisites

- Python 3.10 or later
- A Hugging Face account with access to [`facebook/sam3`](https://huggingface.co/facebook/sam3)
- A Mapbox access token with Geocoding and Tiles API access
- An NVIDIA GPU is strongly recommended. CPU inference is supported but can be very slow.

For GPU installation, install the PyTorch build appropriate to your CUDA driver from the [PyTorch selector](https://pytorch.org/get-started/locally/) before installing this project.

## Local installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Create your local environment file and populate the tokens. Do not commit this file.

```bash
cp .env.example .env
```

Start the local Gradio application. It automatically loads values from `.env`:

```bash
python -m paving_measurement.app
```

Open `http://localhost:7860`. On first use, Transformers downloads and caches the SAM 3 weights.

## Configuration

The application reads all configuration from environment variables. `.env` is a convenient local-development file; Docker and hosted deployments should inject secrets through their platform's secret manager.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `HF_TOKEN` | Yes | — | Hugging Face token allowed to access SAM 3 |
| `MAPBOX_TOKEN` | Yes | — | Mapbox access token |
| `MODEL_ID` | No | `facebook/sam3` | Hugging Face model ID |
| `SEGMENTATION_PROMPT` | No | `asphalt pavement` | Text prompt used for SAM 3 |
| `BATCH_SIZE` | No | `8` | Images per inference batch; adjust for GPU memory |
| `MIN_ZOOM` / `MAX_ZOOM` | No | `14` / `20` | Permitted Mapbox zoom range |
| `DEFAULT_ZOOM` | No | `18` | Initial UI zoom |
| `HOST` / `PORT` | No | `0.0.0.0` / `7860` | Gradio bind address and port |

## Docker

The included image uses a CUDA-enabled PyTorch base. Install the NVIDIA Container Toolkit on the host, add your tokens to `.env`, then run:

```bash
docker compose up --build
```

Or run it directly:

```bash
docker build -t paving-measurement .
docker run --rm --gpus all --env-file .env -p 7860:7860 paving-measurement
```

For a CPU-only deployment, replace the `Dockerfile` base image with an appropriate CPU PyTorch image and remove `gpus: all` from `docker-compose.yml`. Expect substantially longer inference time.

## Modal deployment with FastAPI

`modal_app.py` deploys a FastAPI service instead of the Gradio interface. It uses one NVIDIA A10G GPU, loads SAM 3 once per running container, and persists Hugging Face model files in a Modal Volume to reduce subsequent cold-start downloads. Each `/v1/analyze` request can provide a dynamic comma-separated `prompt`, for example `asphalt pavement, parking lot`.

Install and authenticate the Modal CLI:

```bash
python -m pip install "modal>=1.0"
modal setup
```

Create the Modal secret from your existing local `.env`. The file remains local and is excluded from Git.

```bash
modal secret create paving-measurement-secrets --from-dotenv .env
```

Deploy the API:

```bash
modal deploy modal_app.py
```

The deploy command prints the HTTPS API URL. Use that URL in the following request; FastAPI's interactive OpenAPI documentation is available at `/docs`.

```bash
export MODAL_API_URL="https://your-workspace--paving-measurement-api.modal.run"

curl --location --request POST "$MODAL_API_URL/v1/analyze" \
  --header "Content-Type: application/json" \
  --data '{"address":"1600 Pennsylvania Avenue NW, Washington, DC", "zoom":18}'
```

`POST /v1/satellite` returns the satellite mosaic and coordinates without model inference. `POST /v1/analyze` returns the satellite mosaic, final mask overlay, grid comparison, editable polygon coordinates, ground resolution, measurement summary, and per-grid results. Images are returned as base64 data URLs to keep the API self-contained.

`POST /v1/analyze-polygon` accepts a user-drawn GeoJSON-order polygon ring (`[longitude, latitude]`) instead of an address. It retrieves only the Mapbox tiles that contain the selected area, runs SAM3, and returns the detected object contours in the same geographic coordinate order for drawing directly on a web map. A request is limited to nine source tiles at zoom 16–20 to keep the serverless SAM3 job bounded; draw a smaller area or lower the zoom when that limit is exceeded.

The separate [`frontend/`](frontend/README.md) project provides an address form and browser-side polygon editing. It recalculates square footage as vertices are moved or polygons are removed.

## Parking-stall detection on Modal

`parking_modal_app.py` is a separate serverless GPU deployment for the Hugging Face YOLO model `otmanheddouch/yolov8n-09-19-2026`. It receives the user-drawn areas as GeoJSON-order polygon rings (`[longitude, latitude]`), downloads every source tile covering those rings at the requested zoom, and processes overlapping 2-by-2 tile windows one at a time. Each window is upscaled to the requested inference size (1280px by default), then its centres are converted back to map coordinates. This preserves stall detail for large areas and reduces misses at tile borders. Only centres whose coordinates lie inside a submitted polygon are returned. Overlap duplicates are removed by keeping the highest-confidence centre within two metres.

It uses the existing `paving-measurement-secrets` secret, so it needs `HF_TOKEN` (to download the model) and `MAPBOX_TOKEN` (to download satellite tiles). Deploy it independently:

```bash
modal deploy parking_modal_app.py
```

Then call its `POST /v1/detect-parking-stalls` endpoint. The response's `spots[].coordinates` is directly usable as a map marker position; it is `[longitude, latitude]`, never a bounding box.

```bash
export PARKING_API_URL="https://your-workspace--parking-stall-detection-api.modal.run"

curl --location --request POST "$PARKING_API_URL/v1/detect-parking-stalls" \
  --header "Content-Type: application/json" \
  --data '{
    "zoom": 20,
    "polygons": [[
      [-77.0366, 38.8975],
      [-77.0359, 38.8975],
      [-77.0359, 38.8971],
      [-77.0366, 38.8971]
    ]]
  }'
```

The default request limit is 400 source tiles. For an even larger site, split the drawn region into multiple polygons or lower the zoom; the API returns a clear 400 error rather than starting an unbounded GPU job. `confidence`, `imgsz`, `duplicate_distance_meters`, and `max_tiles` are optional request controls documented in `/docs`.

Modal Web Functions have a 150-second request timeout before returning a redirect to the result URL; use `curl --location` or an HTTP client configured to follow redirects for longer segmentation requests. Keep the generated URL behind appropriate authentication or access controls before sharing it publicly.

## Deployment notes

- Inject `HF_TOKEN` and `MAPBOX_TOKEN` using the hosting platform's secrets facility; never bake them into an image or source file.
- Run one application worker per GPU. The model is deliberately initialized once per worker, not per request.
- Mount a persistent Hugging Face cache volume in production to avoid downloading weights after every rollout.
- Set an upstream request timeout suitable for the selected batch size and GPU. A full multi-grid request evaluates 140 image tiles.
- Place the app behind HTTPS and an authentication layer before exposing it publicly. Mapbox tokens should be URL-restricted where possible.
- Pin and regularly review the base image and Python dependencies before a production release.

## Experiments

[`notebooks/experiments.ipynb`](notebooks/experiments.ipynb) is a compact, repeatable notebook for evaluating grid strategies on a local image. It imports the application helpers rather than duplicating pipeline logic. Start Jupyter from the repository root after installation:

```bash
jupyter lab
```

## Quality checks

```bash
pytest
ruff check .
```

## View an API result locally

Save the response from `/v1/analyze` as `result.json`, then run:

```bash
python test.py result.json
```

The script extracts the satellite image, final overlay, and grid comparison into `outputs/` and opens them together. In a headless environment, save them without opening a window:

```bash
python test.py result.json --no-show
```
