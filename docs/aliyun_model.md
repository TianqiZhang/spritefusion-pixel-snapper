# Qwen Image Edit API (DashScope) - minimal integration notes

This is a condensed reference for `pixel_snapper/qwen.py`. It only includes
the fields needed to build requests and parse responses.

## Endpoints (choose one region)

- Beijing: `https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation`
- Singapore: `https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation`

API keys are region-specific; do not mix keys and endpoints.

## Headers

- `Content-Type: application/json`
- `Authorization: Bearer $DASHSCOPE_API_KEY`

## Request body (single turn)

```json
{
  "model": "qwen-image-edit-plus",
  "input": {
    "messages": [
      {
        "role": "user",
        "content": [
          { "image": "data:image/png;base64,..." },
          { "text": "your edit prompt" }
        ]
      }
    ]
  },
  "parameters": {
    "n": 1,
    "negative_prompt": "optional",
    "prompt_extend": true,
    "watermark": false,
    "size": "512*512",
    "seed": 123
  }
}
```

Notes:
- `content` supports 1-3 images. Each image can be a URL or a data URI.
- `model` is typically `qwen-image-edit-plus` (1-6 outputs). `qwen-image-edit`
  only supports a single output.
- `size` is only supported by the `qwen-image-edit-plus` family and must be
  omitted when `n != 1`.
- The API returns PNG images.

## Response shape (only fields used by code)

```json
{
  "output": {
    "choices": [
      {
        "message": {
          "content": [
            { "image": "https://..." }
          ]
        }
      }
    ]
  }
}
```

`pixel_snapper/qwen.py` reads `output.choices[0].message.content` as a list and
uses the selected index to pull the `image` URL.
