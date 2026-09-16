"""Gradio application entry point."""

from __future__ import annotations

import logging

import gradio as gr
from PIL import Image

from paving_measurement.config import Settings, load_settings
from paving_measurement.mapbox import MapboxClient
from paving_measurement.segmentation import Sam3Segmenter
from paving_measurement.satellite import get_satellite_image


def create_app(settings: Settings) -> gr.Blocks:
    """Create the UI. The model loads once when the application starts."""
    client = MapboxClient(settings.mapbox_token, settings.request_timeout_seconds)
    segmenter = Sam3Segmenter(settings)

    def get_tiles(address: str, latitude_value: float | None, longitude_value: float | None, zoom: float):
        selected_zoom = int(zoom)
        if not settings.min_zoom <= selected_zoom <= settings.max_zoom:
            raise gr.Error(f"Zoom must be between {settings.min_zoom} and {settings.max_zoom}.")
        if address and address.strip():
            latitude, longitude = client.geocode_address(address)
        elif latitude_value is not None and longitude_value is not None:
            latitude, longitude = float(latitude_value), float(longitude_value)
        else:
            raise gr.Error("Enter an address or provide both latitude and longitude.")
        image = get_satellite_image(client, latitude, longitude, selected_zoom)
        status = f"""## Satellite Image Ready

Latitude: `{latitude:.6f}`  
Longitude: `{longitude:.6f}`  
Zoom: `{selected_zoom}`  
Image size: `{image.width} x {image.height}`  
Source: Mapbox satellite tiles
"""
        return image, latitude, longitude, status

    def run_segmentation(image: Image.Image | None, latitude: float | None, longitude: float | None, zoom: float):
        if image is None or latitude is None or longitude is None:
            raise gr.Error("Download satellite tiles before running segmentation.")
        result = segmenter.run(image, float(latitude), int(zoom))
        return result.final_overlay, result.comparison, result.summary, result.grid_rows, result.grid_overlays

    with gr.Blocks(title="SAM 3 Asphalt Segmentation") as demo:
        gr.Markdown("""# SAM 3 Asphalt Segmentation

Mapbox satellite imagery is segmented with SAM 3 to estimate asphalt coverage. Address lookup takes priority over manually entered coordinates.
""")
        gr.Markdown("## 1. Location")
        address = gr.Textbox(label="Address", placeholder="Enter an address, city, road, or landmark")
        with gr.Row():
            latitude = gr.Number(label="Latitude", precision=6)
            longitude = gr.Number(label="Longitude", precision=6)
            zoom = gr.Slider(settings.min_zoom, settings.max_zoom, value=settings.default_zoom, step=1, label="Zoom level")
        get_tiles_button = gr.Button("Get Satellite Tiles", variant="secondary")
        tile_status = gr.Markdown("No satellite image downloaded yet.")
        satellite_image = gr.Image(label="3x3 Mapbox Satellite Tile Mosaic", type="pil")
        gr.Markdown("## 2. Run Segmentation")
        gr.Markdown(f"Prompt: `{settings.prompt}`  \nGrid strategy: `1x1` through `7x7`  \nBatch size: `{settings.batch_size}`")
        run_button = gr.Button("Run Asphalt Segmentation", variant="primary")
        gr.Markdown("## 3. Results")
        with gr.Row():
            final_result = gr.Image(label="Final Unified Asphalt Mask", type="pil")
            comparison = gr.Image(label="Grid Comparison", type="pil")
        summary = gr.Markdown("Run segmentation to see results.")
        grid_table = gr.Dataframe(headers=["Grid", "Masks", "Pixels", "Area (m²)", "Area (ft²)"], datatype=["str", "number", "str", "str", "str"], interactive=False, label="Per-Grid Results")
        grid_gallery = gr.Gallery(label="Individual Grid Outputs", columns=2, rows=4, object_fit="contain")
        get_tiles_button.click(get_tiles, [address, latitude, longitude, zoom], [satellite_image, latitude, longitude, tile_status])
        run_button.click(run_segmentation, [satellite_image, latitude, longitude, zoom], [final_result, comparison, summary, grid_table, grid_gallery])
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    create_app(settings).launch(server_name=settings.host, server_port=settings.port, show_error=True)


if __name__ == "__main__":
    main()
