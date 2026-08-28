# Local_LLM

A desktop GUI (Python/Tkinter) for installing and running LLMs fully locally, offline, on your own PC — powered by `llama.cpp` (with an optional Ollama backend).

## Quick Start

There's no prebuilt executable yet ([Releases](https://github.com/Andrade020/Local_LLM/releases) is currently empty), so run it from source:

```bash
git clone https://github.com/Andrade020/Local_LLM.git
cd Local_LLM
pip install -r requirements.txt
python main.py
```

That's it — `main.py` launches the GUI. On first run, use **File → Open Model** (or set `MODEL_PATH` in `config.env`) to point it at a `.gguf` model file.

## Development Setup

For an isolated environment instead of installing dependencies globally:

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python main.py
```

Requires Python 3.9+ and Tkinter (bundled with Python on Windows/macOS; on Linux install it separately, e.g. `sudo apt-get install python3-tk`). `main.py` checks both dependencies at startup and prints install instructions if either is missing.

### GPU acceleration (optional)

The default install runs on CPU (AVX2). To use a GPU instead:

- **Vulkan** (Intel Iris Xe/Arc, AMD, NVIDIA):
  ```powershell
  $env:CMAKE_ARGS="-DGGML_VULKAN=on"
  pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```
  Then set `n_gpu_layers` > 0 in the app's Settings → Model/Hardware panel.
- **Intel oneAPI / SYCL**: requires the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html); build `llama.cpp` with SYCL support.
- **OpenVINO**: `pip install openvino optimum[openvino]` (requires converting the model to OpenVINO IR/OV format).

See the comments in `requirements.txt` for more detail.

## Features

- Chat interface for local GGUF/GGML models via `llama.cpp`, with an optional Ollama backend
- Automatic hardware detection (CPU cores, RAM, GPU) with acceleration suggestions tailored to what's detected
- Model validation and RAM-requirement checks before loading, so you don't crash your system on a model that's too big
- Persisted settings, response caching, and saved conversation history (browsable/restorable from the app)

## Model Management

- `download_model.bat` downloads a few ready-to-use GGUF models (Llama-2-7B, Mistral-7B, Phi-2, Llama-2-13B) straight into `models/`
- Drop any `.gguf`/`.ggml`/`.bin` model into `models/` and point the app at it — `ModelManager` validates the file, estimates RAM usage from its quantization (Q4/Q5/Q8/etc.), and detects the right chat prompt format automatically
- `ModelManager.suggest_model_size()` recommends a model size range based on your available RAM

## Configuration

Copy `config.example.env` to `config.env` and adjust as needed:

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Path to the `.gguf`/`.ggml` model file to load |
| `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_MAX_TOKENS`, `DEFAULT_REPEAT_PENALTY` | Default inference parameters |
| `CACHE_DIR`, `LOG_LEVEL`, `LOG_FILE` | Cache directory and logging |
| `MAX_RAM_MB` | Cap on RAM usage (`0` = auto-detect) |
| `THEME`, `FONT_SIZE` | UI appearance |

You can also pass `--model`, `--config`, or `--debug` directly to `python main.py` — see `main.py --help`.
