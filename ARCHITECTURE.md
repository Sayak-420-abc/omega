# Architecture
- `backend_onnx.*` — ONNX Runtime inference (SqueezeNet .onnx). At runtime, prints a warning if GPU lib is linked but CUDA runtime isn't present.
- `backend_libtorch.*` — optional TorchScript inference (CPU/CUDA).
- `image_io.*` — stb_image decoding to RGB.
- `infer_api.*` — stable engine shared by CLI + REST.
- `main.cpp` — CLI flags, REST endpoints, labels mapping & JSON.

Data flow: image -> resize 224x224 -> normalize -> NCHW tensor -> forward -> Top-K -> indices/scores (+ labels).
