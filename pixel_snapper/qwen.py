"""Qwen image edit integration for photo-to-pixel-art pre-processing."""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from PIL import Image

from .config import Config, PixelSnapperError

DEFAULT_QWEN_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
DEFAULT_QWEN_MODEL = "qwen-image-edit-plus"
DEFAULT_QWEN_PROMPT = (
    "Transform the subject in this photo into a chibi-style cartoon character - "
    "cute, with a slightly oversized head, large eyes, and simplified features - "
    "then remove all background and render the entire image as a low-resolution "
    "pixel art pattern in the style of Perler or Hama fuse beads. Use a limited "
    "palette of solid, bright colors with no gradients. Each pixel should "
    "represent a single bead (circular or square), arranged on a clear grid. "
    "Keep the composition simple and recognizable, suitable for actual bead "
    "crafting. Preserve key traits (e.g., hairstyle, species, pose, or object "
    "shape) but stylize them in an adorable, minimal chibi pixel form."
)
DEFAULT_QWEN_NEGATIVE_PROMPT = (
    "blurry, low quality, photorealistic, noisy, text, watermark"
)

_FORMAT_TO_MIME = {
    "PNG": "image/png",
    "JPG": "image/jpeg",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "GIF": "image/gif",
}


def maybe_apply_qwen_edit(input_bytes: bytes, config: Config) -> bytes:
    """Optionally run Qwen image edit before pixel snapping."""
    if not config.qwen_enabled:
        return input_bytes
    return _apply_qwen_edit(input_bytes, config)


def _apply_qwen_edit(input_bytes: bytes, config: Config) -> bytes:
    api_key = config.qwen_api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise PixelSnapperError(
            "Qwen image edit enabled, but DASHSCOPE_API_KEY is not set"
        )

    if not (1 <= config.qwen_output_count <= 6):
        raise PixelSnapperError("qwen_output_count must be between 1 and 6")

    if config.qwen_output_count != 1 and config.qwen_size:
        raise PixelSnapperError(
            "qwen_size can only be used when qwen_output_count is 1"
        )

    prompt = (config.qwen_prompt or "").strip() or DEFAULT_QWEN_PROMPT
    negative_prompt = (
        (config.qwen_negative_prompt or "").strip() or DEFAULT_QWEN_NEGATIVE_PROMPT
    )

    image_data = _image_bytes_to_data_uri(input_bytes)
    payload = _build_payload(
        model=config.qwen_model or DEFAULT_QWEN_MODEL,
        image_data=image_data,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_count=config.qwen_output_count,
        prompt_extend=config.qwen_prompt_extend,
        watermark=config.qwen_watermark,
        size=config.qwen_size,
        seed=config.qwen_seed,
    )
    response = _post_json(
        config.qwen_endpoint or DEFAULT_QWEN_ENDPOINT,
        api_key,
        payload,
        timeout=config.qwen_timeout,
    )
    image_url = _extract_image_url(response, config.qwen_output_index)
    return _download_image(image_url, timeout=config.qwen_timeout)


def _image_bytes_to_data_uri(input_bytes: bytes) -> str:
    with Image.open(io.BytesIO(input_bytes)) as img:
        fmt = (img.format or "PNG").upper()
        if fmt not in _FORMAT_TO_MIME:
            fmt = "PNG"
            buffer = io.BytesIO()
            img.save(buffer, format=fmt)
            input_bytes = buffer.getvalue()
    mime_type = _FORMAT_TO_MIME[fmt]
    encoded = base64.b64encode(input_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_payload(
    model: str,
    image_data: str,
    prompt: str,
    negative_prompt: Optional[str],
    output_count: int,
    prompt_extend: bool,
    watermark: bool,
    size: Optional[str],
    seed: Optional[int],
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "n": output_count,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if negative_prompt:
        parameters["negative_prompt"] = negative_prompt
    if size:
        parameters["size"] = size
    if seed is not None:
        parameters["seed"] = seed

    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_data},
                        {"text": prompt},
                    ],
                }
            ]
        },
        "parameters": parameters,
    }


def _post_json(
    endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        message = _format_http_error(body, exc)
        raise PixelSnapperError(message) from exc
    except urllib.error.URLError as exc:
        raise PixelSnapperError(f"Qwen request failed: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise PixelSnapperError("Qwen response was not valid JSON") from exc


def _format_http_error(body: str, exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {}
    code = payload.get("code") or payload.get("error_code")
    message = payload.get("message") or payload.get("error_message")
    detail = f"{code}: {message}" if code or message else body.strip()
    detail = detail or "Unknown error"
    return f"Qwen request failed (HTTP {exc.code}): {detail}"


def _extract_image_url(response: Dict[str, Any], output_index: int) -> str:
    try:
        content = response["output"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PixelSnapperError("Qwen response missing image content") from exc

    if not isinstance(content, list) or not content:
        raise PixelSnapperError("Qwen response contained no images")
    if output_index < 0 or output_index >= len(content):
        raise PixelSnapperError(
            f"qwen_output_index {output_index} out of range for {len(content)} images"
        )

    item = content[output_index]
    image_url = item.get("image") if isinstance(item, dict) else None
    if not image_url:
        raise PixelSnapperError("Qwen response did not include an image URL")
    return image_url


def _download_image(image_url: str, timeout: int) -> bytes:
    try:
        with urllib.request.urlopen(image_url, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise PixelSnapperError(f"Failed to download Qwen image: {exc}") from exc
