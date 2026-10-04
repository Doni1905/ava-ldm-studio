# Verified environment setup

Use Python 3.11 and Node 22.12 or newer. No virtual environment is shipped in this repository.

Follow the setup commands and limitations at the top of [README.md](../README.md). On Windows activate `.venv\Scripts\Activate.ps1`; on macOS/Linux activate `source .venv/bin/activate`. Use the regular PyPI torch wheel on macOS, the CPU wheel index on Linux without a GPU, and the appropriate PyTorch GPU index only if you have a supported GPU.

The requirements now include aiohttp (the actual API framework) and matplotlib (benchmark reporting), and use versions tested on Python 3.11. No Hugging Face login is needed for the synthetic benchmark or public Whisper weights. Private/gated datasets may require separately authorized access.

Run `python -m pytest tests -q`, `npm ci`, `npm run typecheck`, `npm run lint` and `npm run build`. Run the Python API and Vite frontend in separate terminals. `/health` and `/analyze` are implemented; `/docs` is not.

The Android module needs JDK 17 and Android SDK 35; its APK/device behavior remains unverified. A trained acoustic dialect checkpoint is not shipped. First audio use downloads Whisper weights; after that local inference can use the cache. Browser Web Speech may use the vendor cloud.
