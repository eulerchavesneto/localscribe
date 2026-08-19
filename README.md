# 🎙️ Whisper Transcriber

A desktop app that transcribes audio and video with [OpenAI's Whisper](https://github.com/openai/whisper) — running **entirely on your machine**. No API keys, no uploads, no third-party services.

Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for real GPU acceleration and [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for a modern interface.

---

## Features

- **Actually uses your GPU.** CUDA is detected at startup, precision is chosen automatically to fit your VRAM, and when the GPU *can't* be used the app tells you exactly why instead of silently crawling along on CPU.
- **Batch processing.** Queue any number of local files and URLs, watch each one's status, and cancel mid-run.
- **Local files and online video.** Drag and drop media onto the window, or paste links from any site [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports. Playlists can be expanded into individual videos.
- **Multiple export formats.** `.txt`, `.srt`, `.vtt` and `.json` — pick any combination.
- **15 languages** with auto-detection, plus optional translation to English.
- **Voice activity detection (VAD)** skips silence, which speeds things up and reduces Whisper's tendency to hallucinate text during quiet passages.
- **No FFmpeg needed.** Decoding goes through PyAV.
- **Light and dark themes**, remembered between sessions along with the rest of your settings.

---

## Requirements

- **Python 3.9+**
- **An NVIDIA GPU** (optional) for acceleration. Without one everything still works on CPU, just slower.
- **A JavaScript runtime** (optional) — only needed for downloading online video. See [below](#javascript-runtime).

---

## Installation

```bash
git clone https://github.com/eulerchavesneto/transcription-semAI.git
cd transcription-semAI

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Then run it:

```bash
python transcription_semAI.py
```

The first transcription downloads the selected Whisper model (a few hundred MB to ~3 GB depending on the model). That happens once and is cached afterwards.

---

## JavaScript runtime

**This only matters if you want to transcribe online videos.** Local files need nothing extra.

YouTube requires clients to solve a JavaScript challenge before it hands out media URLs. Without a runtime that yt-dlp accepts, downloads fail with confusing errors like `Requested format is not available` or `HTTP Error 403: Forbidden`.

Install **one** of these:

| Runtime | Minimum version | |
|---|---|---|
| Node.js | **22.0.0** | https://nodejs.org |
| Deno | 2.3.0 | https://deno.com |
| Bun | 1.2.11 | https://bun.sh |

> **Having the runtime installed is not enough — it must meet the minimum version.** An older Node.js (18, for example) is detected by the system but rejected by yt-dlp, which then silently degrades to a JS-less mode where many videos have no downloadable audio. The app checks the actual version on startup and shows a badge in the header, so you'll know immediately whether yours qualifies.

---

## Usage

1. **Add sources** — drag files onto the window, click **Add Files**, or paste links with **Add URLs**.
2. **Pick a model.** `large-v3-turbo` is the default and the best speed/accuracy trade-off for most people. Drop to `small` or `base` if you're on CPU or short on VRAM.
3. **Choose an output folder.** Required when processing more than one source, or any URL.
4. **Select your formats** and hit **Transcribe**.

The estimate next to the model dropdown shows how much VRAM the current combination needs and warns you before you start something that won't fit.

### Choosing a model

| Model | VRAM (float16) | VRAM (int8_float16) | Notes |
|---|---|---|---|
| `tiny` / `base` | ~0.4–0.5 GB | ~0.3 GB | Fast, noticeably less accurate |
| `small` | ~0.7 GB | ~0.45 GB | Good CPU option |
| `medium` | ~2.0 GB | ~1.1 GB | |
| `distil-large-v3` | ~1.6 GB | ~0.95 GB | Near-large quality, lighter |
| **`large-v3-turbo`** | ~2.2 GB | ~1.2 GB | **Recommended default** |
| `large-v2` / `large-v3` | ~3.7 GB | ~2.1 GB | Most accurate, heaviest |

Measured on an RTX 3050 4GB with `beam_size=5`. If a model doesn't fit in VRAM it spills into system memory and gets dramatically slower — the app warns you first.

---

## Troubleshooting

**"Requested format is not available" / "HTTP Error 403"**
Almost always a missing or outdated JavaScript runtime — check the badge in the header and see [above](#javascript-runtime). If your runtime is fine, update yt-dlp: `pip install -U yt-dlp`.

**The header says CPU but I have an NVIDIA card**
Click **Why no GPU?** for the specific reason. The usual fix is installing the CUDA libraries, which do *not* require the CUDA Toolkit:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

**Transcription is extremely slow on GPU**
The model probably doesn't fit in VRAM and is spilling into system memory. Switch precision to `int8_float16`, or use a smaller model.

**Repeated or hallucinated text**
Keep the voice filter (VAD) enabled — it skips silent stretches, which is where Whisper tends to invent text.

---

## How it works

```
Sources (files / URLs)
        │
        ├── URLs → yt-dlp ──→ audio track in a temp folder
        │                     (per-task dir, always cleaned up)
        │
        └── Local files ─────┐
                             ▼
                   faster-whisper (CTranslate2)
                   PyAV decode → 16 kHz mono → GPU/CPU inference
                             │
                             ▼
                 .txt / .srt / .vtt / .json
```

The `openai-whisper` backend is also selectable if installed, mainly for comparing output against the reference implementation. It's slower, uses roughly twice the memory, and does require FFmpeg.

---

## License

Personal project, provided as-is. Whisper models are released by OpenAI under the MIT license.
