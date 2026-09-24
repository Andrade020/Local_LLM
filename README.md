# Local_LLM

A desktop GUI (Python/Tkinter) for running LLMs fully locally, offline, on your own PC. It's powered by `llama.cpp`: the app starts the official `llama-server` binary in the background and chats with it.

Using the official binary instead of `llama-cpp-python` means new model architectures work right away (e.g. Qwen3.6 MoE), CUDA works on older NVIDIA GPUs (GTX 10xx / Pascal), and you get MoE expert offload to the CPU (`--n-cpu-moe`), which is what lets a 35B model run fast on a 3 GB GPU.

## What's new in v2.0

This is a new version of the app. Version 1.0 ran models in-process through `llama-cpp-python`, CPU only. v2.0 changes:

- **New backend:** the app now drives `llama-server` from the official `llama.cpp` releases, with GPU offload and MoE expert offload. `llama-cpp-python` is no longer a dependency.
- **Real chat:** the full conversation history is sent using the model's own chat template. v1.0 sent only the last message as a raw prompt.
- **Thinking mode** for Qwen3.x models, with the reasoning shown separately from the answer.
- **Tokens/s** in the status bar, and a working **Stop** button.
- **Hardware settings in `config.env`:** `LLAMA_SERVER_PATH`, `N_GPU_LAYERS`, `N_CPU_MOE`, `N_CTX`, `N_THREADS`, `ENABLE_THINKING`. The `DEFAULT_*` inference values are now applied too.
- **Fixed memory check:** v1.0 overestimated RAM for i-quants, and it ignored mmap and VRAM.
- **Recommended model** changed from Llama-2 to Qwen3.6-35B-A3B.
- **`config.env` is no longer tracked:** it holds machine-specific paths. Copy it from `config.example.env`.

## Quick Start

1. Download `llama.cpp` from [Releases](https://github.com/ggml-org/llama.cpp/releases) and unzip it:
   - NVIDIA GTX 10xx (Pascal): `llama-bXXXX-bin-win-cuda-12.4-x64.zip` **and** `cudart-llama-bin-win-cuda-12.4-x64.zip` (CUDA 13 no longer supports Pascal)
   - Newer NVIDIA: the `cuda-13.x` builds; AMD/Intel: `win-vulkan`; no GPU: `win-cpu`
2. Download a GGUF model (see below).
3. Set up and run:

```bash
git clone https://github.com/Andrade020/Local_LLM.git
cd Local_LLM
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy config.example.env config.env   # then edit LLAMA_SERVER_PATH and MODEL_PATH
python main.py                        # or double-click iniciar.bat
```

The model in `MODEL_PATH` is loaded automatically when the app starts. You can also load one from **Arquivo → Carregar Modelo**.

Requires Python 3.9+ and Tkinter (bundled with Python on Windows/macOS; on Linux install it separately, e.g. `sudo apt-get install python3-tk`).

## Recommended model: Qwen3.6-35B-A3B

A MoE model with 35B total parameters and only 3B active per token (MMLU-Pro 85, GPQA 86). Tested on a Ryzen 5 1600 + GTX 1060 3GB + 16 GB single-channel DDR4:

| Model | Config | Generation |
|---|---|---|
| [Qwen3.6-35B-A3B UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (12-13 GB) | `N_GPU_LAYERS=99`, `N_CPU_MOE=40`, `N_THREADS=12`, `N_CTX=32768` | **~11-12 tokens/s** |
| Qwen3.5-4B Q4_K_M (2.5 GB) | CPU only | ~4 tokens/s |

On low-RAM-bandwidth machines, a MoE model with a small number of active parameters is much faster than a dense 7-9B model and much smarter.

## Features

- Chat with full conversation history using the model's own chat template
- Thinking mode (Qwen3.x): the reasoning is shown in gray, separate from the answer, and can be turned on/off in **Configurações**
- Tokens/s shown in the status bar after each answer
- Model validation and memory check before loading (accounts for mmap and GPU VRAM)
- Response caching and conversation export (txt/json/md)

## Configuration

Copy `config.example.env` to `config.env` and adjust as needed:

| Variable | Purpose |
|---|---|
| `LLAMA_SERVER_PATH` | Path to `llama-server.exe` |
| `MODEL_PATH` | Path to the `.gguf` model file to load |
| `N_GPU_LAYERS` | Layers offloaded to the GPU (`99` = all, `0` = CPU only) |
| `N_CPU_MOE` | MoE models: number of layers whose experts stay on the CPU. With little VRAM, start at the model's layer count and lower it while it still fits |
| `N_CTX`, `N_THREADS` | Context size and CPU threads |
| `LLAMA_SERVER_EXTRA_ARGS` | Any extra `llama-server` flags |
| `ENABLE_THINKING` | Default for thinking mode (`true`/`false`) |
| `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_MAX_TOKENS`, `DEFAULT_REPEAT_PENALTY` | Default inference parameters |
| `CACHE_DIR`, `LOG_LEVEL`, `LOG_FILE` | Cache directory and logging |

The `llama-server` output is written to `logs/llama-server.log`. You can also pass `--model`, `--config`, or `--debug` directly to `python main.py` (see `main.py --help`).
