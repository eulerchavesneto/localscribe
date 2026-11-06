# 🎙️ LocalScribe

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
git clone https://github.com/eulerchavesneto/localscribe.git
cd localscribe

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Then run it:

```bash
python localscribe.py
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

## Transcribing a whole course (command line)

`course_transcribe.py` batch-transcribes an entire online course hosted on **Panda Video** (currently set up for the Desec Academy, `academy.desecsecurity.com`) without watching or exporting anything by hand. It reuses the same faster-whisper engine as the app and has no GUI. It only works for videos you have legitimate access to, and the output is for personal study.

**1. Build the manifest.** Log in to the course site in your browser and open any lesson. Open the console (`F12` → Console), paste the contents of [`course_manifest.js`](course_manifest.js) and press Enter. It reads each lesson page with your own session (about one per second), then downloads a `manifest.json` with module, title and video id for every lesson. Keep the tab open until it finishes.

**2. List the modules.**

```bash
python course_transcribe.py manifest.json --list
```

**3. Transcribe.** Pick modules by the numbers shown by `--list`, or omit `--modules` to do everything:

```bash
python course_transcribe.py manifest.json --modules 4 5 --model small
python course_transcribe.py manifest.json --timestamps        # whole course, with [hh:mm:ss]
```

| Option | Default | |
|---|---|---|
| `--modules N [N ...]` | all | Module numbers from `--list` |
| `--model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, … |
| `--language` | `pt` | Language code passed to Whisper |
| `--timestamps` | off | Prefix every segment with `[hh:mm:ss]` |
| `--out` | `transcricoes` | Output folder |
| `--referer` | Desec Academy | Site of the course; Panda only serves video with that `Referer` |

Output, ready to feed to an AI assistant:

```
transcricoes/
├── 04_introducao-ao-penetration-testing.md     # whole module in one file
└── 04_introducao-ao-penetration-testing/
    ├── 001_o-que-e-pentest.md                  # one file per lesson
    └── 002_...
```

It is resumable: lessons that already have a `.md` are skipped, so you can stop and run it again. A lesson that fails is reported and the batch continues. Each video is downloaded at the lowest quality (the audio is the same), transcribed, then deleted. `manifest.json` and `transcricoes/` are git-ignored because they are course content.

> On CPU a full course takes days. Start with a few modules and the `small` model, or use an NVIDIA GPU with `--model large-v3-turbo`.

---

## Troubleshooting

**"Requested format is not available"**
Almost always a missing or outdated JavaScript runtime — check the badge in the header and see [above](#javascript-runtime).

**"unable to download video data: HTTP Error 403: Forbidden"**
This one is *not* a network problem, and retrying doesn't help. Extraction succeeds — yt-dlp obtains the format URLs — but YouTube then refuses to serve the media, because it now demands a [Proof-of-Origin (PO) token](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide) on Google Video Server requests for its default player clients.

LocalScribe works around this by requesting clients that are exempt from the GVS token requirement, in order: `web_embedded`, `mweb`, `tv`, `web_safari` (see `YOUTUBE_PLAYER_CLIENTS`). If 403s come back, YouTube has probably widened enforcement — try reordering that list, or install a PO token provider such as [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider).

**"Sign in to confirm you're not a bot"**
Bot detection, usually triggered by many downloads in quick succession. It clears on its own; retrying immediately won't help.

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
