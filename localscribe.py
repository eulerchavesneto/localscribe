# coding: utf-8
"""LocalScribe — a private audio/video transcription desktop app.

Transcribes local media files or online videos with OpenAI's Whisper models,
running entirely on your machine. Nothing is ever uploaded to a third-party API.

Highlights
----------
- `faster-whisper` backend (CTranslate2): roughly 4x faster and far lighter on
  VRAM than the reference `openai-whisper`, with no PyTorch dependency.
- GPU acceleration that actually works: CUDA detection, automatic precision
  selection based on available VRAM, and a clear diagnosis when the GPU cannot
  be used.
- Voice Activity Detection (VAD) to skip silence and reduce hallucinations.
- Real per-file progress instead of an indeterminate spinner.
- Export to .txt, .srt, .vtt and .json.
- Cancellable batches.
- Resilient downloader: playlist expansion, automatic retry on intermittent
  HTTP 403s, and error messages a human can act on.
- FFmpeg is optional — decoding goes through PyAV.
"""

import ctypes
import gc
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --- CUDA DLL bootstrap (must run BEFORE ctranslate2 is imported) ------------
#
# The `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels drop their DLLs inside
# site-packages but never add them to the PATH. CTranslate2 loads them through
# LoadLibrary, which searches the PATH, so `os.add_dll_directory` alone is not
# enough — the PATH entry is what makes the difference.

CUDA_DLL_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc")
REQUIRED_CUDA_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")


def _nvidia_package_bin(pkg: str) -> Optional[Path]:
    """Locate the `bin` directory of an `nvidia-*` wheel, venv or global."""
    try:
        spec = importlib.util.find_spec(f"nvidia.{pkg}")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        candidate = Path(list(spec.submodule_search_locations)[0]) / "bin"
        if candidate.is_dir():
            return candidate
    fallback = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / pkg / "bin"
    return fallback if fallback.is_dir() else None


def _bootstrap_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    for pkg in CUDA_DLL_PACKAGES:
        bin_dir = _nvidia_package_bin(pkg)
        if bin_dir is None:
            continue
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_bootstrap_cuda_dlls()


# --- Dependency check (no automatic installation) ----------------------------
#
# v1 used to `pip install torch` on its own, which on Windows pulls the CPU-only
# wheel and made "GPU acceleration" silently impossible. Installation is now
# explicit and documented in requirements.txt.

REQUIRED_MODULES = {
    "faster_whisper": "faster-whisper",
    "yt_dlp": "yt-dlp",
    "yt_dlp_ejs": "yt-dlp-ejs",
    "tkinterdnd2": "tkinterdnd2",
    "customtkinter": "customtkinter",
}


def check_dependencies() -> None:
    missing = [pkg for mod, pkg in REQUIRED_MODULES.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing Dependencies",
        "The following packages are not installed:\n\n"
        f"    {', '.join(missing)}\n\n"
        "Install them with:\n\n"
        f"    {Path(sys.executable).name} -m pip install -r requirements.txt\n\n"
        f"Interpreter in use:\n{sys.executable}",
    )
    sys.exit(1)


check_dependencies()

import customtkinter as ctk  # noqa: E402
import yt_dlp  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from tkinterdnd2 import TkinterDnD  # noqa: E402


# --- JavaScript runtime detection --------------------------------------------
#
# YouTube only hands out media URLs to clients that can solve a JavaScript
# challenge (the "n challenge"). That needs two things: the yt-dlp-ejs package
# (the solver) and a JavaScript runtime on the system. Without both, downloads
# fail misleadingly — sometimes "Requested format is not available", sometimes
# "HTTP Error 403: Forbidden" — and the failure is intermittent, because yt-dlp
# falls back to JS-less YouTube clients that occasionally still return
# something.
#
# Crucially, "a runtime is installed" is NOT the same as "a runtime yt-dlp can
# use": each runtime carries a minimum supported version. Checking only for the
# executable (the old `shutil.which` approach) reports success for, say, Node 18
# while yt-dlp rejects it and silently degrades to JS-less extraction. The
# minimum versions are read straight from yt-dlp so this never drifts out of
# sync with the installed release.

# Every runtime yt-dlp knows about is enabled; yt-dlp defaults to Deno alone.
JS_RUNTIMES = {"deno": {}, "node": {}, "bun": {}, "quickjs": {}}


@dataclass
class JsRuntimeStatus:
    name: str = ""        # usable runtime, or "" when there is none
    problem: str = ""     # why there is none, ready to show to the user

    @property
    def ok(self) -> bool:
        return bool(self.name)


def _format_version(version: Sequence[int]) -> str:
    return ".".join(str(part) for part in version)


def detect_js_runtime() -> JsRuntimeStatus:
    """Find a JavaScript runtime that yt-dlp will actually accept.

    Runs the `--version` probe of every runtime yt-dlp supports, so an installed
    but too-old runtime is reported as such instead of passing as usable.
    """
    from yt_dlp.globals import supported_js_runtimes

    outdated: List[str] = []
    for name, runtime_cls in supported_js_runtimes.value.items():
        if runtime_cls is None:
            continue
        try:
            info = runtime_cls().info
        except Exception:  # a broken runtime must never take the app down
            continue
        if info is None:
            continue  # not installed
        if info.supported:
            return JsRuntimeStatus(name=name)
        minimum = _format_version(getattr(runtime_cls, "MIN_SUPPORTED_VERSION", ()))
        outdated.append(f"{info.name} {info.version} (needs {minimum} or newer)")

    if outdated:
        return JsRuntimeStatus(problem=(
            "A JavaScript runtime was found, but it is too old for yt-dlp:\n\n"
            + "\n".join(f"    • {item}" for item in outdated)
            + "\n\nUpgrade it, or install one of the alternatives below."))
    return JsRuntimeStatus(problem="No JavaScript runtime was found.")


JS_RUNTIME_HELP = (
    "YouTube requires a JavaScript runtime to release its media URLs. Without "
    "one, downloads fail with errors like 'Requested format is not available' "
    "or 'HTTP Error 403'.\n\n"
    "Install any one of these and restart the app — it detects them "
    "automatically:\n\n"
    "    • Node.js 22 or newer — https://nodejs.org\n"
    "    • Deno 2.3 or newer — https://deno.com\n"
    "    • Bun 1.2.11 or newer — https://bun.sh\n\n"
    "Transcribing local files keeps working normally without a runtime."
)


# --- Device detection --------------------------------------------------------

@dataclass
class DeviceInfo:
    device: str = "cpu"      # "cuda" or "cpu"
    gpu_name: str = ""
    total_vram_mb: int = 0
    problem: str = ""        # why the GPU could not be used

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def vram_known(self) -> bool:
        """False when the VRAM query failed and the budget is a guess."""
        return self.total_vram_mb > 0

    @property
    def usable_vram_mb(self) -> int:
        """Total VRAM minus headroom for the desktop and browser."""
        return max(self.total_vram_mb - 600, 0)

    def describe(self) -> str:
        if not self.is_cuda:
            cores = os.cpu_count() or "?"
            return f"CPU ({cores} cores)" + (f" — {self.problem}" if self.problem else "")
        if self.vram_known:
            return f"GPU: {self.gpu_name} — {self.total_vram_mb / 1024:.1f} GB VRAM"
        return f"GPU: {self.gpu_name} — VRAM unknown"


def _run_hidden(args: List[str], timeout: int = 8) -> str:
    """Run a command without flashing a console window on Windows."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                             creationflags=flags)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _query_gpu() -> Tuple[str, int]:
    out = _run_hidden(["nvidia-smi", "--query-gpu=name,memory.total",
                       "--format=csv,noheader,nounits"])
    if not out:
        return "CUDA GPU", 0
    name, _, mem = out.splitlines()[0].partition(",")
    try:
        return name.strip(), int(mem.strip())
    except ValueError:
        return name.strip(), 0


def detect_device() -> DeviceInfo:
    """Work out whether CUDA is usable and, if not, explain why."""
    import ctranslate2

    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # pragma: no cover - environment dependent
        return DeviceInfo(problem=f"CTranslate2 could not query CUDA: {exc}")

    if cuda_count < 1:
        return DeviceInfo(problem="no NVIDIA GPU detected")

    # CTranslate2 only loads cuBLAS/cuDNN on the first inference call. Without
    # this probe the user would discover a missing DLL halfway through a batch.
    if sys.platform == "win32":
        for dll in REQUIRED_CUDA_DLLS:
            try:
                ctypes.WinDLL(dll)
            except OSError:
                return DeviceInfo(problem=(
                    f"GPU detected, but {dll} is missing. "
                    "Install: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"))

    name, vram = _query_gpu()
    return DeviceInfo(device="cuda", gpu_name=name, total_vram_mb=vram)


# --- Model catalogue ---------------------------------------------------------
#
# VRAM in MiB, measured on an RTX 3050 4GB (weights + activations, beam_size=5).
# Values marked ~ are extrapolated from the measured ones.

MODELS: Dict[str, Dict[str, int]] = {
    "tiny":            {"float16": 400,  "int8_float16": 300},   # ~
    "base":            {"float16": 500,  "int8_float16": 350},   # ~
    "small":           {"float16": 727,  "int8_float16": 455},   # measured
    "medium":          {"float16": 2079, "int8_float16": 1087},  # measured
    "distil-large-v3": {"float16": 1600, "int8_float16": 950},   # ~
    "large-v3-turbo":  {"float16": 2215, "int8_float16": 1199},  # measured
    "large-v2":        {"float16": 3735, "int8_float16": 2071},  # ~ (same as v3)
    "large-v3":        {"float16": 3735, "int8_float16": 2071},  # measured
}

DEFAULT_MODEL = "large-v3-turbo"
AUTO_PRECISION = "Auto"

CPU_COMPUTE_TYPES = ["int8", "float32"]
CUDA_COMPUTE_TYPES = ["float16", "int8_float16", "int8", "float32"]


def auto_compute_type(model: str, dev: DeviceInfo) -> str:
    """Pick the fastest precision that still fits in the available VRAM.

    When the VRAM budget is unknown the conservative choice wins: an unknown
    budget is not a large one, and guessing float16 here is how a 4 GB card ends
    up thrashing halfway through a batch.
    """
    if not dev.is_cuda:
        return "int8"
    if not dev.vram_known:
        return "int8_float16"
    budget = dev.usable_vram_mb
    needs = MODELS.get(model, MODELS[DEFAULT_MODEL])
    if needs["float16"] <= budget:
        return "float16"
    return "int8_float16"  # best effort; the UI has already warned


def vram_estimate_mb(model: str, compute_type: str) -> int:
    needs = MODELS.get(model)
    if not needs:
        return 0
    return needs.get(compute_type, needs["float16"])


# --- Output formats ----------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str


def _timestamp(seconds: float, sep: str) -> str:
    ms = max(int(round(seconds * 1000)), 0)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def to_txt(segments: List[Segment], text: str) -> str:
    return text + "\n"


def to_srt(segments: List[Segment], text: str) -> str:
    blocks = []
    for index, seg in enumerate(segments, start=1):
        blocks.append(f"{index}\n"
                      f"{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}\n"
                      f"{seg.text}\n")
    return "\n".join(blocks)


def to_vtt(segments: List[Segment], text: str) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}\n"
                      f"{seg.text}\n")
    return "\n".join(blocks)


def to_json(segments: List[Segment], text: str) -> str:
    payload = {
        "text": text,
        "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                     for s in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


WRITERS: Dict[str, Callable[[List[Segment], str], str]] = {
    "txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json,
}


# --- Transcription backends --------------------------------------------------

class Cancelled(Exception):
    """Raised when the user aborts processing."""


@dataclass
class TranscriptionResult:
    text: str
    segments: List[Segment]
    language: str
    duration: float


class FasterWhisperBackend:
    """Default backend: CTranslate2, no PyTorch."""

    name = "faster-whisper"

    def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None
        self._key: Tuple[str, str, str] = ("", "", "")

    def load(self, model_name: str, device: str, compute_type: str) -> None:
        key = (model_name, device, compute_type)
        if self._model is not None and key == self._key:
            return
        self.unload()  # never hold two models in VRAM at once
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._key = key

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._key = ("", "", "")
            gc.collect()

    def transcribe(self, path: Path, language: Optional[str], task: str, vad: bool,
                   on_progress: Callable[[float], None],
                   cancel: threading.Event) -> TranscriptionResult:
        assert self._model is not None, "load() must be called before transcribe()"
        segment_iter, info = self._model.transcribe(
            str(path),
            language=language,
            task=task,
            beam_size=5,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # fewer repetitions on long audio
        )
        collected: List[Segment] = []
        total = info.duration or 0.0
        for seg in segment_iter:
            if cancel.is_set():
                raise Cancelled()
            collected.append(Segment(seg.start, seg.end, seg.text.strip()))
            if total > 0:
                on_progress(min(seg.end / total * 100.0, 100.0))
        on_progress(100.0)
        text = " ".join(s.text for s in collected).strip()
        return TranscriptionResult(text, collected, info.language or "", total)


class OpenAIWhisperBackend:
    """Legacy backend. Only offered in the UI when `openai-whisper` is present."""

    name = "openai-whisper"

    def __init__(self) -> None:
        self._model: Any = None
        self._key: Tuple[str, str] = ("", "")

    @staticmethod
    def available() -> bool:
        return (importlib.util.find_spec("whisper") is not None
                and importlib.util.find_spec("torch") is not None)

    def load(self, model_name: str, device: str, compute_type: str) -> None:
        import whisper
        key = (model_name, device)
        if self._model is not None and key == self._key:
            return
        self.unload()
        self._model = whisper.load_model(model_name, device=device)
        self._key = key

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._key = ("", "")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(self, path: Path, language: Optional[str], task: str, vad: bool,
                   on_progress: Callable[[float], None],
                   cancel: threading.Event) -> TranscriptionResult:
        assert self._model is not None, "load() must be called before transcribe()"
        result = self._model.transcribe(
            str(path), language=language, task=task,
            fp16=(self._key[1] == "cuda"),  # fp16 only makes sense on GPU
        )
        if cancel.is_set():
            raise Cancelled()
        segments = [Segment(s["start"], s["end"], s["text"].strip())
                    for s in result.get("segments", [])]
        on_progress(100.0)
        duration = segments[-1].end if segments else 0.0
        return TranscriptionResult(result["text"].strip(), segments,
                                   result.get("language", ""), duration)


BACKENDS = {cls.name: cls for cls in (FasterWhisperBackend, OpenAIWhisperBackend)}


# --- Media download (yt-dlp) --------------------------------------------------

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

# A URL copied from inside a playlist carries `list=...` even when the intent is
# just that one video — the normal case for anything copied out of "Watch
# later". Personal playlists are private and need a login to enumerate; since
# this app downloads without credentials, offering to expand them would only
# produce an error.
PLAYLIST_ID_PATTERN = re.compile(r"[?&]list=([^&]+)")
PRIVATE_PLAYLIST_IDS = frozenset({"WL", "LL"})  # Watch later, Liked videos


def is_expandable_playlist(url: str) -> bool:
    match = PLAYLIST_ID_PATTERN.search(url)
    return bool(match) and match.group(1) not in PRIVATE_PLAYLIST_IDS


YDL_ERROR_HINTS = (
    (("requested format is not available", "no video formats", "requested format"),
     "YouTube did not release any audio track for this video.\n\n"
     "The usual cause is a missing JavaScript runtime, needed to solve "
     "YouTube's signature challenge. Install Node.js 22+ (https://nodejs.org) "
     "or Deno and restart the app — it detects them automatically.\n\n"
     "If a runtime is already installed, update yt-dlp:\n"
     "    pip install -U yt-dlp"),
    (("http error 403", "forbidden", "sign in to confirm", "not a bot"),
     "YouTube refused the download after several attempts.\n\n"
     "This is usually temporary — wait a few minutes and try again. If it "
     "persists, update yt-dlp:\n    pip install -U yt-dlp"),
    (("private video", "members-only", "join this channel"),
     "Private or members-only video. This app downloads without signing in, "
     "so there is no way to reach it."),
    (("video unavailable", "removed by the uploader"),
     "Video unavailable or removed."),
    (("age-restricted", "age restricted", "inappropriate"),
     "Age-restricted video. This app downloads without signing in, so there is "
     "no way to reach it."),
    (("unable to download webpage", "getaddrinfo", "timed out", "connection"),
     "Network failure while contacting the server. Check your connection."),
)


def friendly_download_error(exc: Exception) -> str:
    raw = str(exc)
    low = raw.lower()
    for needles, hint in YDL_ERROR_HINTS:
        if any(needle in low for needle in needles):
            return hint
    return raw.strip() or "Unknown download failure."


class QuietLogger:
    """Trim yt-dlp noise without hiding diagnostics.

    `debug`/`info` are dropped (that is the progress feed, which already drives
    the UI progress bar). Warnings and errors still reach the terminal,
    deduplicated: they are what say things like "no JavaScript runtime found",
    without which an unavailable-format error is impossible to diagnose.
    Deduplication stops a batch of N tasks from repeating the same line 2N times.
    """

    def __init__(self) -> None:
        self._seen: set = set()

    def _emit(self, level: str, msg: str) -> None:
        if msg not in self._seen:
            self._seen.add(msg)
            print(f"[yt-dlp {level}] {msg}", file=sys.stderr)

    def debug(self, msg: str) -> None: pass
    def info(self, msg: str) -> None: pass
    def warning(self, msg: str) -> None: self._emit("warning", msg)
    def error(self, msg: str) -> None: self._emit("error", msg)


def base_ydl_opts() -> Dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": False,  # warnings are the diagnosis; QuietLogger filters them
        "noprogress": True,    # progress drives the UI bar, not stdout
        "logger": QuietLogger(),
        "js_runtimes": JS_RUNTIMES,
    }


def build_ydl_opts(temp_dir: Path, hook: Optional[Callable] = None) -> Dict[str, Any]:
    """Download the audio track only, with no post-processing.

    There is deliberately no FFmpegExtractAudio postprocessor: `faster-whisper`
    decodes m4a/webm directly through PyAV and resamples to 16 kHz mono in
    memory. That removes the FFmpeg requirement and avoids writing a huge WAV to
    disk.
    """
    opts: Dict[str, Any] = {
        **base_ydl_opts(),
        "format": "bestaudio/best",
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),  # id is short and always valid
        "noplaylist": True,  # playlists are expanded explicitly from the UI
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "nocheckcertificate": False,
    }
    if hook is not None:
        opts["progress_hooks"] = [hook]
    return opts


def downloaded_filepath(ydl: "yt_dlp.YoutubeDL", info: Dict[str, Any]) -> Path:
    """The real path of the downloaded file.

    `prepare_filename()` guesses the name; `requested_downloads` is what yt-dlp
    actually wrote — they differ whenever the extension changes.
    """
    downloads = info.get("requested_downloads")
    if downloads and downloads[0].get("filepath"):
        return Path(downloads[0]["filepath"])
    return Path(ydl.prepare_filename(info))


# YouTube intermittently returns 403 on media URLs it signed itself: the same
# call fails and, seconds later, succeeds. Retrying the same URL does not help —
# the whole extraction has to run again to get a freshly signed URL, which is
# why yt-dlp's own `retries` does not cover this case.
#
# The waits are deliberately long. After several downloads in a row YouTube
# starts rate-limiting the IP, and a 2-3 second pause achieves nothing: the
# throttling window needs real time to expire.
DOWNLOAD_RETRY_DELAYS = (5, 20, 45)
DOWNLOAD_ATTEMPTS = len(DOWNLOAD_RETRY_DELAYS) + 1

RETRIABLE_DOWNLOAD_ERRORS = (
    "http error 403", "forbidden", "unable to download video data",
    "requested format is not available", "unable to download webpage",
    "too many requests", "http error 429",
)


def is_retriable_download_error(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(needle in low for needle in RETRIABLE_DOWNLOAD_ERRORS)


def download_media(url: str, temp_dir: Path, hook: Optional[Callable],
                   cancel: threading.Event,
                   on_retry: Optional[Callable[[int, int], None]] = None,
                   ) -> Tuple[Path, Dict[str, Any]]:
    """Download the audio track of `url`, re-extracting on transient failures."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(build_ydl_opts(temp_dir, hook)) as ydl:
                info = ydl.extract_info(url, download=True)
                if not isinstance(info, dict):
                    raise ValueError("yt-dlp returned no metadata for this URL.")
                return downloaded_filepath(ydl, info), info
        except Exception as exc:
            if (cancel.is_set() or attempt == DOWNLOAD_ATTEMPTS
                    or not is_retriable_download_error(exc)):
                raise
            delay = DOWNLOAD_RETRY_DELAYS[attempt - 1]
            if on_retry is not None:
                on_retry(attempt + 1, delay)
            # `cancel.wait` rather than `time.sleep`: during a 45 s pause the
            # Cancel button still has to respond.
            if cancel.wait(delay):
                raise
    raise AssertionError("unreachable: the loop above always returns or raises")


def safe_output_stem(title: str, fallback: str) -> str:
    stem = yt_dlp.utils.sanitize_filename(title or fallback, restricted=False)
    stem = stem.strip().rstrip(".")  # Windows rejects names ending in a dot
    return stem[:120] or fallback


def expand_playlist(url: str) -> List[str]:
    """Resolve a playlist URL into its video URLs, downloading nothing."""
    opts = {**base_ydl_opts(), "extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    # Some playlist-like URLs resolve to nothing; without this guard the caller
    # would surface an opaque AttributeError instead of a usable message.
    if not isinstance(info, dict):
        raise ValueError("This link did not resolve to a readable playlist.")
    urls = []
    for entry in info.get("entries") or []:
        if not entry:
            continue
        target = entry.get("url") or entry.get("webpage_url") or entry.get("id")
        if target and not URL_PATTERN.match(target):
            target = f"https://www.youtube.com/watch?v={target}"
        if target:
            urls.append(target)
    return urls


# --- Settings persistence ----------------------------------------------------

SETTINGS_FILE = Path.home() / ".localscribe.json"
# Preferences written before the project was renamed; read once so an upgrade
# does not silently reset the user's settings.
LEGACY_SETTINGS_FILE = Path.home() / ".transcription_semai.json"


def load_settings() -> Dict[str, Any]:
    for path in (SETTINGS_FILE, LEGACY_SETTINGS_FILE):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def save_settings(data: Dict[str, Any]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # preferences are a nicety; they must never break the app


# --- Task state --------------------------------------------------------------

class Status:
    QUEUED = "queued"
    WORKING = "working"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Statuses that mean "this task produced its output; do not redo it".
TERMINAL_SUCCESS = {Status.DONE}


@dataclass
class Task:
    source: Any                 # Path for local files, str for URLs
    display: str
    status: str = Status.QUEUED
    detail: str = "Queued"      # human-readable status shown in the list
    text: str = ""
    # Kept so "Save As…" can still export .srt/.vtt/.json after the batch ends.
    segments: List["Segment"] = field(default_factory=list)
    error: str = ""
    saved: List[Path] = field(default_factory=list)

    @property
    def is_url(self) -> bool:
        return isinstance(self.source, str) and bool(URL_PATTERN.match(self.source))

    @property
    def is_done(self) -> bool:
        return self.status in TERMINAL_SUCCESS


# --- Theme -------------------------------------------------------------------

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

# Every colour is a (light, dark) pair so both appearance modes stay legible.
COLORS = {
    "bg":            ("#f2f4f7", "#16181d"),
    "surface":       ("#ffffff", "#1e2128"),
    "surface_alt":   ("#f7f8fa", "#24272f"),
    "border":        ("#dfe3e8", "#31353f"),
    "text":          ("#1c2024", "#e8eaed"),
    "text_muted":    ("#5c6470", "#9aa3b0"),
    "accent":        ("#2563eb", "#3b82f6"),
    "accent_hover":  ("#1d4ed8", "#60a5fa"),
    "success":       ("#15803d", "#4ade80"),
    "success_bg":    ("#dcfce7", "#14351f"),
    "warning":       ("#b45309", "#fbbf24"),
    "warning_bg":    ("#fef3c7", "#3a2c0a"),
    "danger":        ("#b91c1c", "#f87171"),
    "danger_bg":     ("#fee2e2", "#3b1414"),
    "info_bg":       ("#dbeafe", "#15294d"),
    "muted_bg":      ("#eef0f3", "#2a2e37"),
}

# Badge styling per task status: (background, foreground).
STATUS_BADGE = {
    Status.QUEUED:    ("muted_bg", "text_muted"),
    Status.WORKING:   ("info_bg", "accent"),
    Status.DONE:      ("success_bg", "success"),
    Status.FAILED:    ("danger_bg", "danger"),
    Status.CANCELLED: ("warning_bg", "warning"),
}


def color(name: str) -> Tuple[str, str]:
    return COLORS[name]


# --- Reusable dialogs --------------------------------------------------------

class ModalDialog(ctk.CTkToplevel):
    """A themed modal dialog, so the app never falls back to native chrome."""

    ICONS = {
        "info": ("ⓘ", "accent"),
        "success": ("✓", "success"),
        "warning": ("⚠", "warning"),
        "error": ("✕", "danger"),
        "question": ("?", "accent"),
    }

    def __init__(self, parent, title: str, message: str, kind: str = "info",
                 buttons: Sequence[str] = ("OK",), destructive: bool = False):
        super().__init__(parent)
        self.result: Optional[str] = None
        self._buttons = list(buttons)

        self.title(title)
        self.configure(fg_color=color("bg"))
        self.resizable(False, False)
        self.transient(parent)

        glyph, tone = self.ICONS.get(kind, self.ICONS["info"])

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(22, 16))

        header = ctk.CTkFrame(body, fg_color="transparent")
        header.pack(fill="x", anchor="w")
        ctk.CTkLabel(header, text=glyph, text_color=color(tone),
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(header, text=title, text_color=color("text"),
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")

        ctk.CTkLabel(body, text=message, justify="left", anchor="w",
                     wraplength=460, text_color=color("text_muted"),
                     font=ctk.CTkFont(size=12)).pack(fill="x", pady=(14, 0))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=24, pady=(0, 20))

        # Rightmost button is the default/confirming action.
        for index, label in enumerate(reversed(self._buttons)):
            is_primary = index == 0
            if is_primary and destructive:
                fg, hover = color("danger"), color("danger")
            elif is_primary:
                fg, hover = color("accent"), color("accent_hover")
            else:
                fg, hover = "transparent", color("muted_bg")
            ctk.CTkButton(
                actions, text=label, width=104, height=34, corner_radius=8,
                fg_color=fg, hover_color=hover,
                text_color="#ffffff" if is_primary else color("text"),
                border_width=0 if is_primary else 1,
                border_color=color("border"),
                font=ctk.CTkFont(size=12, weight="bold" if is_primary else "normal"),
                command=lambda value=label: self._choose(value),
            ).pack(side="right", padx=(8, 0))

        self.protocol("WM_DELETE_WINDOW", lambda: self._choose(self._buttons[0]))
        self.bind("<Escape>", lambda _e: self._choose(self._buttons[0]))
        self.bind("<Return>", lambda _e: self._choose(self._buttons[-1]))

        self._center_on(parent)
        # grab_set() only works once the window is actually mapped.
        self.wait_visibility()
        self.grab_set()
        self.focus_force()

    def _center_on(self, parent) -> None:
        self.update_idletasks()
        width, height = self.winfo_width(), self.winfo_height()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 3
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _choose(self, value: str) -> None:
        self.result = value
        self.grab_release()
        self.destroy()

    @classmethod
    def ask(cls, parent, title: str, message: str, kind: str = "info",
            buttons: Sequence[str] = ("OK",), destructive: bool = False) -> Optional[str]:
        dialog = cls(parent, title, message, kind, buttons, destructive)
        parent.wait_window(dialog)
        return dialog.result


def show_info(parent, title: str, message: str) -> None:
    ModalDialog.ask(parent, title, message, "info")


def show_success(parent, title: str, message: str) -> None:
    ModalDialog.ask(parent, title, message, "success")


def show_warning(parent, title: str, message: str) -> None:
    ModalDialog.ask(parent, title, message, "warning")


def show_error(parent, title: str, message: str) -> None:
    ModalDialog.ask(parent, title, message, "error")


def ask_yes_no(parent, title: str, message: str, yes: str = "Yes", no: str = "No",
               destructive: bool = False) -> bool:
    return ModalDialog.ask(parent, title, message, "question",
                           buttons=(no, yes), destructive=destructive) == yes


# --- Task list row -----------------------------------------------------------

class TaskRow(ctk.CTkFrame):
    """One row of the source list: label, status badge and a remove button."""

    def __init__(self, parent, task_id: str, task: Task,
                 on_remove: Callable[[str], None],
                 on_select: Callable[[str], None]):
        super().__init__(parent, fg_color=color("surface"), corner_radius=8,
                         border_width=1, border_color=color("border"))
        self.task_id = task_id
        self._on_select = on_select
        self._selected = False

        self.grid_columnconfigure(1, weight=1)

        icon = "🔗" if task.is_url else "🎞"
        self.icon_label = ctk.CTkLabel(self, text=icon, width=26,
                                       font=ctk.CTkFont(size=13))
        self.icon_label.grid(row=0, column=0, padx=(10, 4), pady=9)

        self.name_label = ctk.CTkLabel(self, text=self._elide(task.display), anchor="w",
                                       text_color=color("text"),
                                       font=ctk.CTkFont(size=12))
        self.name_label.grid(row=0, column=1, sticky="ew", padx=(2, 10), pady=9)

        self.badge = ctk.CTkLabel(self, text=task.detail, corner_radius=6,
                                  width=118, height=23,
                                  font=ctk.CTkFont(size=11, weight="bold"))
        self.badge.grid(row=0, column=2, padx=(0, 8), pady=9)

        self.remove_button = ctk.CTkButton(
            self, text="✕", width=26, height=26, corner_radius=6,
            fg_color="transparent", hover_color=color("danger_bg"),
            text_color=color("text_muted"), font=ctk.CTkFont(size=13),
            command=lambda: on_remove(task_id))
        self.remove_button.grid(row=0, column=3, padx=(0, 8), pady=9)

        self.set_status(task)
        # Children swallow clicks, so every one of them has to forward the event.
        for widget in (self, self.icon_label, self.name_label, self.badge):
            widget.bind("<Button-1>", self._clicked)

    @staticmethod
    def _elide(text: str, limit: int = 78) -> str:
        return text if len(text) <= limit else text[:limit - 1] + "…"

    def _clicked(self, _event) -> None:
        self._on_select(self.task_id)

    def set_status(self, task: Task) -> None:
        bg_key, fg_key = STATUS_BADGE.get(task.status, STATUS_BADGE[Status.QUEUED])
        self.badge.configure(text=self._elide(task.detail, 22),
                             fg_color=color(bg_key), text_color=color(fg_key))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.configure(border_color=color("accent") if selected else color("border"),
                       border_width=2 if selected else 1)

    def set_enabled(self, enabled: bool) -> None:
        self.remove_button.configure(state="normal" if enabled else "disabled")


# --- Application -------------------------------------------------------------

class DnDCTk(ctk.CTk, TkinterDnD.DnDWrapper):
    """CustomTkinter root with tkinterdnd2 drag-and-drop support bolted on."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class LocalScribeApp:
    LANGUAGES = {
        "Auto-detect": None, "English": "en", "Portuguese": "pt", "Spanish": "es",
        "French": "fr", "German": "de", "Italian": "it", "Japanese": "ja",
        "Chinese": "zh", "Russian": "ru", "Korean": "ko", "Arabic": "ar",
        "Dutch": "nl", "Polish": "pl", "Turkish": "tr",
    }

    MEDIA_EXTENSIONS = ("*.mp3 *.mp4 *.m4a *.wav *.avi *.mkv *.mov *.flac "
                        "*.ogg *.opus *.webm *.aac *.wma")

    def __init__(self, root: DnDCTk):
        self.root = root
        self.root.title("LocalScribe")
        self.root.geometry("1120x860")
        self.root.minsize(960, 720)
        self.root.configure(fg_color=color("bg"))

        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings.get("appearance", "system"))

        # Hardware and runtime probing both shell out to external binaries, so
        # they run off the main thread; the window must not wait on them.
        self.device_info = DeviceInfo(problem="detecting...")
        self.js_runtime = JsRuntimeStatus()
        self.probe_complete = False

        self.backend: Any = FasterWhisperBackend()
        self.output_directory = Path(self.settings.get("output_dir", ""))
        self.event_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.tasks: Dict[str, Task] = {}
        self.rows: Dict[str, TaskRow] = {}
        self.task_order: List[str] = []
        self.task_counter = 0
        self.selected_task: Optional[str] = None
        self.is_processing = False
        self.is_expanding = False
        self.cancel_event = threading.Event()
        self.ffmpeg_path = Path()

        self._setup_ffmpeg_environment()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._pump_events)
        threading.Thread(target=self._probe_environment, daemon=True).start()

    # -- environment probing -------------------------------------------------

    def _probe_environment(self) -> None:
        """Detect GPU and JS runtime off the main thread (both spawn processes)."""
        device = detect_device()
        runtime = detect_js_runtime()
        self.event_queue.put(("probe_done", (device, runtime)))

    def _on_probe_done(self, payload: Tuple[DeviceInfo, JsRuntimeStatus]) -> None:
        self.device_info, self.js_runtime = payload
        self.probe_complete = True

        is_cuda = self.device_info.is_cuda
        self.device_badge.configure(
            text=f"{'⚡' if is_cuda else '🖥'}  {self.device_info.describe()}",
            fg_color=color("success_bg" if is_cuda else "warning_bg"),
            text_color=color("success" if is_cuda else "warning"))
        if is_cuda:
            self.gpu_help_button.pack_forget()
        else:
            self.gpu_help_button.pack(side="left", padx=(0, 8),
                                      before=self.appearance_button)

        allowed = CUDA_COMPUTE_TYPES if is_cuda else CPU_COMPUTE_TYPES
        self.precision_menu.configure(values=[AUTO_PRECISION] + allowed)
        if self.precision_var.get() not in [AUTO_PRECISION] + allowed:
            self.precision_var.set(AUTO_PRECISION)

        if self.js_runtime.ok:
            self.runtime_badge.configure(
                text=f"JS: {self.js_runtime.name}",
                fg_color=color("success_bg"), text_color=color("success"))
        else:
            self.runtime_badge.configure(
                text="JS runtime missing",
                fg_color=color("warning_bg"), text_color=color("warning"))

        self.start_button.configure(state="normal")
        self._set_status("Ready")
        self._refresh_estimate()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._build_header()

        self.body = ctk.CTkFrame(self.root, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(4, 0))
        self.body.grid_rowconfigure(0, weight=1)
        self.body.grid_columnconfigure(0, weight=1)

        self._build_main_view()
        self._build_results_view()
        self._build_footer()
        self.show_main_view()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self.root, fg_color=color("surface"), corner_radius=0,
                              height=64)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkFrame(header, fg_color="transparent")
        title.grid(row=0, column=0, sticky="w", padx=(20, 0), pady=12)
        ctk.CTkLabel(title, text="🎙", font=ctk.CTkFont(size=22)).pack(side="left",
                                                                      padx=(0, 10))
        text_block = ctk.CTkFrame(title, fg_color="transparent")
        text_block.pack(side="left")
        ctk.CTkLabel(text_block, text="LocalScribe", anchor="w",
                     text_color=color("text"),
                     font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(text_block, text="Local speech-to-text — nothing leaves your machine",
                     anchor="w", text_color=color("text_muted"),
                     font=ctk.CTkFont(size=11)).pack(anchor="w")

        badges = ctk.CTkFrame(header, fg_color="transparent")
        badges.grid(row=0, column=2, sticky="e", padx=(0, 16), pady=12)

        self.runtime_badge = ctk.CTkLabel(
            badges, text="checking…", corner_radius=6, height=26, width=104,
            fg_color=color("muted_bg"), text_color=color("text_muted"),
            font=ctk.CTkFont(size=11, weight="bold"))
        self.runtime_badge.pack(side="left", padx=(0, 8))
        self.runtime_badge.bind("<Button-1>", lambda _e: self._show_runtime_help())

        self.device_badge = ctk.CTkLabel(
            badges, text="🔎  Detecting hardware…", corner_radius=6, height=26,
            fg_color=color("muted_bg"), text_color=color("text_muted"),
            font=ctk.CTkFont(size=11, weight="bold"))
        self.device_badge.pack(side="left", padx=(0, 8))

        self.gpu_help_button = ctk.CTkButton(
            badges, text="Why no GPU?", width=96, height=26, corner_radius=6,
            fg_color="transparent", border_width=1, border_color=color("border"),
            text_color=color("text"), hover_color=color("muted_bg"),
            font=ctk.CTkFont(size=11), command=self._show_gpu_help)
        # Only shown once probing proves the GPU is unusable.
        self.gpu_help_button.pack_forget()

        self.appearance_button = ctk.CTkButton(
            badges, text="◐", width=32, height=26, corner_radius=6,
            fg_color="transparent", border_width=1, border_color=color("border"),
            text_color=color("text"), hover_color=color("muted_bg"),
            font=ctk.CTkFont(size=13), command=self._toggle_appearance)
        self.appearance_button.pack(side="left")

    def _card(self, parent, title: str, subtitle: str = "") -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=color("surface"), corner_radius=12,
                            border_width=1, border_color=color("border"))
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(13, 0))
        ctk.CTkLabel(head, text=title, anchor="w", text_color=color("text"),
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        if subtitle:
            ctk.CTkLabel(head, text=subtitle, anchor="w",
                         text_color=color("text_muted"),
                         font=ctk.CTkFont(size=11)).pack(side="left", padx=(10, 0))
        return card

    def _build_main_view(self) -> None:
        self.main_view = ctk.CTkScrollableFrame(self.body, fg_color="transparent")
        self.main_view.grid(row=0, column=0, sticky="nsew")
        self.main_view.grid_columnconfigure(0, weight=1)

        self._build_sources_card()
        self._build_settings_card()
        self._build_output_card()

    def _build_sources_card(self) -> None:
        card = self._card(self.main_view, "Sources",
                          "drag files anywhere onto this list")
        card.grid(row=0, column=0, sticky="ew", pady=(8, 12))
        card.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(card, fg_color="transparent")
        toolbar.pack(fill="x", padx=16, pady=(10, 8))

        self.add_files_button = self._button(toolbar, "＋  Add Files",
                                             self._browse_files, primary=True)
        self.add_files_button.pack(side="left")
        self.add_urls_button = self._button(toolbar, "🔗  Add URLs", self._add_urls_dialog)
        self.add_urls_button.pack(side="left", padx=8)
        self.clear_done_button = self._button(toolbar, "Clear Completed",
                                              self._clear_completed)
        self.clear_done_button.pack(side="right")
        self.clear_button = self._button(toolbar, "Clear All", self._clear_all)
        self.clear_button.pack(side="right", padx=8)

        self.task_list = ctk.CTkScrollableFrame(
            card, fg_color=color("surface_alt"), corner_radius=8, height=228)
        self.task_list.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.task_list.grid_columnconfigure(0, weight=1)

        self.empty_label = ctk.CTkLabel(
            self.task_list,
            text="\n\nNo sources yet\n\nDrop media files here, or use the buttons above.\n",
            text_color=color("text_muted"), font=ctk.CTkFont(size=12))
        self.empty_label.grid(row=0, column=0, pady=26)

        # Drop targets must be registered on the concrete tk widgets.
        for widget in (self.task_list, self.empty_label):
            widget.drop_target_register("DND_Files")
            widget.dnd_bind("<<Drop>>", self._handle_drop)

    def _build_settings_card(self) -> None:
        card = self._card(self.main_view, "Settings")
        card.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=16, pady=(10, 14))
        for col in (1, 3):
            grid.grid_columnconfigure(col, weight=1)

        # Row 0 — model and precision
        self._field_label(grid, "Model", 0, 0)
        self.model_var = tk.StringVar(value=self.settings.get("model", DEFAULT_MODEL))
        self.model_menu = ctk.CTkOptionMenu(
            grid, variable=self.model_var, values=list(MODELS), width=180, height=32,
            corner_radius=8, font=ctk.CTkFont(size=12),
            fg_color=color("surface_alt"), button_color=color("border"),
            button_hover_color=color("muted_bg"), text_color=color("text"),
            dropdown_fg_color=color("surface"), dropdown_text_color=color("text"),
            command=lambda _v: self._refresh_estimate())
        self.model_menu.grid(row=0, column=1, sticky="w", padx=(0, 22), pady=6)

        self._field_label(grid, "Precision", 0, 2)
        self.precision_var = tk.StringVar(
            value=self.settings.get("compute_type", AUTO_PRECISION))
        self.precision_menu = ctk.CTkOptionMenu(
            grid, variable=self.precision_var, values=[AUTO_PRECISION], width=160,
            height=32, corner_radius=8, font=ctk.CTkFont(size=12),
            fg_color=color("surface_alt"), button_color=color("border"),
            button_hover_color=color("muted_bg"), text_color=color("text"),
            dropdown_fg_color=color("surface"), dropdown_text_color=color("text"),
            command=lambda _v: self._refresh_estimate())
        self.precision_menu.grid(row=0, column=3, sticky="w", pady=6)

        self.estimate_label = ctk.CTkLabel(grid, text="", anchor="w",
                                           text_color=color("text_muted"),
                                           font=ctk.CTkFont(size=11, weight="bold"))
        self.estimate_label.grid(row=0, column=4, sticky="w", padx=(18, 0), pady=6)

        # Row 1 — language and backend
        self._field_label(grid, "Language", 1, 0)
        self.language_var = tk.StringVar(
            value=self.settings.get("language", "Auto-detect"))
        if self.language_var.get() not in self.LANGUAGES:
            self.language_var.set("Auto-detect")
        self.language_menu = ctk.CTkOptionMenu(
            grid, variable=self.language_var, values=list(self.LANGUAGES), width=180,
            height=32, corner_radius=8, font=ctk.CTkFont(size=12),
            fg_color=color("surface_alt"), button_color=color("border"),
            button_hover_color=color("muted_bg"), text_color=color("text"),
            dropdown_fg_color=color("surface"), dropdown_text_color=color("text"))
        self.language_menu.grid(row=1, column=1, sticky="w", padx=(0, 22), pady=6)

        self._field_label(grid, "Backend", 1, 2)
        backends = [FasterWhisperBackend.name]
        if OpenAIWhisperBackend.available():
            backends.append(OpenAIWhisperBackend.name)
        self.backend_var = tk.StringVar(value=FasterWhisperBackend.name)
        self.backend_menu = ctk.CTkOptionMenu(
            grid, variable=self.backend_var, values=backends, width=160, height=32,
            corner_radius=8, font=ctk.CTkFont(size=12),
            fg_color=color("surface_alt"), button_color=color("border"),
            button_hover_color=color("muted_bg"), text_color=color("text"),
            dropdown_fg_color=color("surface"), dropdown_text_color=color("text"),
            command=lambda _v: self._refresh_estimate())
        self.backend_menu.grid(row=1, column=3, sticky="w", pady=6)

        # Row 2 — toggles
        toggles = ctk.CTkFrame(grid, fg_color="transparent")
        toggles.grid(row=2, column=0, columnspan=5, sticky="w", pady=(10, 2))

        self.translate_var = tk.BooleanVar(value=False)
        self.translate_switch = self._switch(toggles, "Translate to English",
                                             self.translate_var)
        self.translate_switch.pack(side="left", padx=(0, 26))

        self.vad_var = tk.BooleanVar(value=self.settings.get("vad", True))
        self.vad_switch = self._switch(toggles, "Voice filter (skip silence)",
                                       self.vad_var)
        self.vad_switch.pack(side="left")

        # Row 3 — output formats
        formats_row = ctk.CTkFrame(grid, fg_color="transparent")
        formats_row.grid(row=3, column=0, columnspan=5, sticky="w", pady=(12, 0))
        ctk.CTkLabel(formats_row, text="Save as", text_color=color("text_muted"),
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 14))

        self.format_vars: Dict[str, tk.BooleanVar] = {}
        self.format_boxes: List[ctk.CTkCheckBox] = []
        saved_formats = self.settings.get("formats", ["txt"])
        for fmt in WRITERS:
            var = tk.BooleanVar(value=fmt in saved_formats)
            self.format_vars[fmt] = var
            box = ctk.CTkCheckBox(
                formats_row, text=f".{fmt}", variable=var, width=64, checkbox_width=18,
                checkbox_height=18, corner_radius=5, border_width=2,
                font=ctk.CTkFont(size=12), text_color=color("text"),
                fg_color=color("accent"), hover_color=color("accent_hover"),
                border_color=color("border"))
            box.pack(side="left", padx=(0, 10))
            self.format_boxes.append(box)

    def _build_output_card(self) -> None:
        card = self._card(self.main_view, "Output folder",
                          "required for batches and URLs")
        card.grid(row=2, column=0, sticky="ew", pady=(0, 14))

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(10, 14))

        self.output_var = tk.StringVar(
            value=str(self.output_directory) if self.output_directory.is_dir()
            else "No folder selected")
        self.output_entry = ctk.CTkEntry(
            row, textvariable=self.output_var, height=34, corner_radius=8,
            font=ctk.CTkFont(size=12), fg_color=color("surface_alt"),
            border_color=color("border"), text_color=color("text_muted"))
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.output_entry.configure(state="readonly")

        self.browse_button = self._button(row, "Browse…", self._select_output_directory)
        self.browse_button.pack(side="left")
        self.open_folder_button = self._button(row, "Open", self._open_output_dir)
        self.open_folder_button.pack(side="left", padx=(8, 0))

    def _build_results_view(self) -> None:
        self.results_view = ctk.CTkFrame(self.body, fg_color="transparent")
        self.results_view.grid_rowconfigure(1, weight=1)
        self.results_view.grid_columnconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(self.results_view, fg_color="transparent")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(8, 10))
        self._button(toolbar, "←  Back", self.show_main_view).pack(side="left")
        self._button(toolbar, "Copy", self._copy_to_clipboard).pack(side="right")
        self._button(toolbar, "Save As…", self._save_current).pack(side="right", padx=8)
        self._button(toolbar, "Open Folder", self._open_output_dir).pack(side="right")

        self.result_list = ctk.CTkScrollableFrame(
            self.results_view, fg_color=color("surface"), corner_radius=12,
            border_width=1, border_color=color("border"), width=290)
        self.result_list.grid(row=1, column=0, sticky="nsw", pady=(0, 14))
        self.result_list.grid_columnconfigure(0, weight=1)

        detail = ctk.CTkFrame(self.results_view, fg_color=color("surface"),
                              corner_radius=12, border_width=1,
                              border_color=color("border"))
        detail.grid(row=1, column=1, sticky="nsew", padx=(12, 0), pady=(0, 14))
        detail.grid_rowconfigure(1, weight=1)
        detail.grid_columnconfigure(0, weight=1)

        self.result_title = ctk.CTkLabel(detail, text="", anchor="w",
                                         text_color=color("text"),
                                         font=ctk.CTkFont(size=13, weight="bold"))
        self.result_title.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))

        self.result_text = ctk.CTkTextbox(
            detail, corner_radius=8, wrap="word", font=ctk.CTkFont(size=13),
            fg_color=color("surface_alt"), text_color=color("text"),
            border_width=1, border_color=color("border"))
        self.result_text.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.result_text.configure(state="disabled")

        self.result_rows: Dict[str, TaskRow] = {}

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self.root, fg_color=color("surface"), corner_radius=0,
                              height=76)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_propagate(False)
        footer.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(footer, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=20, pady=14)
        left.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(left, text="Detecting hardware…", anchor="w",
                                         text_color=color("text_muted"),
                                         font=ctk.CTkFont(size=12))
        self.status_label.grid(row=0, column=0, sticky="ew")

        bars = ctk.CTkFrame(left, fg_color="transparent")
        bars.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ctk.CTkLabel(bars, text="Current", text_color=color("text_muted"),
                     font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 6))
        self.task_bar = ctk.CTkProgressBar(bars, width=190, height=7, corner_radius=4,
                                           progress_color=color("accent"),
                                           fg_color=color("muted_bg"))
        self.task_bar.pack(side="left", padx=(0, 18))
        self.task_bar.set(0)

        ctk.CTkLabel(bars, text="Overall", text_color=color("text_muted"),
                     font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 6))
        self.total_bar = ctk.CTkProgressBar(bars, width=190, height=7, corner_radius=4,
                                            progress_color=color("success"),
                                            fg_color=color("muted_bg"))
        self.total_bar.pack(side="left")
        self.total_bar.set(0)

        actions = ctk.CTkFrame(footer, fg_color="transparent")
        actions.grid(row=0, column=1, sticky="e", padx=(0, 20), pady=14)

        self.cancel_button = ctk.CTkButton(
            actions, text="Cancel", width=100, height=42, corner_radius=9,
            fg_color="transparent", border_width=1, border_color=color("border"),
            text_color=color("text"), hover_color=color("danger_bg"),
            font=ctk.CTkFont(size=12), state="disabled",
            command=self._cancel_processing)
        self.cancel_button.pack(side="left", padx=(0, 10))

        self.start_button = ctk.CTkButton(
            actions, text="Transcribe", width=176, height=42, corner_radius=9,
            fg_color=color("accent"), hover_color=color("accent_hover"),
            text_color="#ffffff", font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled", command=self._start_processing)
        self.start_button.pack(side="left")

    # -- small widget factories ----------------------------------------------

    def _button(self, parent, text: str, command: Callable,
                primary: bool = False) -> ctk.CTkButton:
        if primary:
            return ctk.CTkButton(
                parent, text=text, command=command, height=32, corner_radius=8,
                fg_color=color("accent"), hover_color=color("accent_hover"),
                text_color="#ffffff", font=ctk.CTkFont(size=12, weight="bold"))
        return ctk.CTkButton(
            parent, text=text, command=command, height=32, corner_radius=8,
            fg_color="transparent", border_width=1, border_color=color("border"),
            text_color=color("text"), hover_color=color("muted_bg"),
            font=ctk.CTkFont(size=12))

    def _switch(self, parent, text: str, variable: tk.BooleanVar) -> ctk.CTkSwitch:
        return ctk.CTkSwitch(
            parent, text=text, variable=variable, font=ctk.CTkFont(size=12),
            text_color=color("text"), progress_color=color("accent"),
            button_color="#ffffff", fg_color=color("muted_bg"),
            switch_width=40, switch_height=20)

    def _field_label(self, parent, text: str, row: int, column: int) -> None:
        ctk.CTkLabel(parent, text=text, anchor="w", text_color=color("text_muted"),
                     font=ctk.CTkFont(size=12)).grid(
            row=row, column=column, sticky="w", padx=(0, 12), pady=6)

    # -- navigation ----------------------------------------------------------

    def show_main_view(self) -> None:
        self.results_view.grid_remove()
        self.main_view.grid(row=0, column=0, sticky="nsew")

    def show_results_view(self) -> None:
        self.main_view.grid_remove()
        self.results_view.grid(row=0, column=0, sticky="nsew")

    def _toggle_appearance(self) -> None:
        current = ctk.get_appearance_mode()
        new_mode = "light" if current.lower() == "dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        self.settings["appearance"] = new_mode

    def _show_gpu_help(self) -> None:
        show_info(self.root, "GPU acceleration",
                  f"Reason: {self.device_info.problem or 'unknown'}\n\n"
                  "Requirements for GPU acceleration:\n\n"
                  "    1. An NVIDIA card with an up-to-date driver.\n"
                  "    2. The CUDA libraries, installable via pip:\n\n"
                  "        pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n\n"
                  "Neither the CUDA Toolkit nor PyTorch is required.")

    def _show_runtime_help(self) -> None:
        if self.js_runtime.ok:
            show_success(self.root, "JavaScript runtime",
                         f"Using '{self.js_runtime.name}'. YouTube downloads are "
                         "fully supported.")
            return
        show_warning(self.root, "JavaScript runtime missing",
                     f"{self.js_runtime.problem}\n\n{JS_RUNTIME_HELP}")

    # -- source list ---------------------------------------------------------

    def _add_tasks(self, sources: List[Any]) -> None:
        added = 0
        existing = {str(task.source) for task in self.tasks.values()}
        for source in sources:
            if str(source) in existing:
                continue  # silently skip exact duplicates
            existing.add(str(source))
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
            display = source.name if isinstance(source, Path) else str(source)
            task = Task(source=source, display=display)
            self.tasks[task_id] = task
            self.task_order.append(task_id)
            row = TaskRow(self.task_list, task_id, task, self._remove_task,
                          self._select_task)
            self.rows[task_id] = row
            added += 1
        self._relayout_task_list()
        if added:
            self._set_status(f"{added} source(s) added — {len(self.tasks)} in the list.")

    def _relayout_task_list(self) -> None:
        self.empty_label.grid_remove()
        for index, task_id in enumerate(self.task_order):
            self.rows[task_id].grid(row=index, column=0, sticky="ew", padx=8,
                                    pady=(6 if index == 0 else 3, 3))
        if not self.task_order:
            self.empty_label.grid(row=0, column=0, pady=26)

    def _remove_task(self, task_id: str) -> None:
        if self.is_processing:
            return
        row = self.rows.pop(task_id, None)
        if row is not None:
            row.destroy()
        self.tasks.pop(task_id, None)
        if task_id in self.task_order:
            self.task_order.remove(task_id)
        if self.selected_task == task_id:
            self.selected_task = None
        self._relayout_task_list()

    def _select_task(self, task_id: str) -> None:
        self.selected_task = task_id
        for tid, row in self.rows.items():
            row.set_selected(tid == task_id)

    def _clear_all(self) -> None:
        if self.is_processing:
            return
        for row in self.rows.values():
            row.destroy()
        self.rows.clear()
        self.tasks.clear()
        self.task_order.clear()
        self.selected_task = None
        self.task_counter = 0
        self._relayout_task_list()
        self._set_status("List cleared.")

    def _clear_completed(self) -> None:
        if self.is_processing:
            return
        done = [tid for tid in self.task_order if self.tasks[tid].is_done]
        for task_id in done:
            self._remove_task(task_id)
        self._set_status(f"{len(done)} completed source(s) removed."
                         if done else "No completed sources to remove.")

    def _handle_drop(self, event) -> None:
        raw = event.data
        paths = re.findall(r"\{([^}]+)\}", raw) if "{" in raw else raw.split()
        if paths:
            self._add_tasks([Path(p) for p in paths])

    def _browse_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select media files",
            filetypes=(("Supported media", self.MEDIA_EXTENSIONS), ("All files", "*.*")))
        if paths:
            self._add_tasks([Path(p) for p in paths])

    def _add_urls_dialog(self) -> None:
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Add URLs")
        dialog.geometry("600x400")
        dialog.configure(fg_color=color("bg"))
        dialog.transient(self.root)

        ctk.CTkLabel(dialog, text="Add URLs", anchor="w", text_color=color("text"),
                     font=ctk.CTkFont(size=15, weight="bold")).pack(
            fill="x", padx=24, pady=(20, 2))
        ctk.CTkLabel(dialog, text="One per line. Any site yt-dlp supports works, "
                                  "not just YouTube.", anchor="w",
                     text_color=color("text_muted"),
                     font=ctk.CTkFont(size=11)).pack(fill="x", padx=24, pady=(0, 12))

        box = ctk.CTkTextbox(dialog, corner_radius=8, font=ctk.CTkFont(size=12),
                             fg_color=color("surface"), text_color=color("text"),
                             border_width=1, border_color=color("border"))
        box.pack(fill="both", expand=True, padx=24)
        box.focus_set()

        actions = ctk.CTkFrame(dialog, fg_color="transparent")
        actions.pack(fill="x", padx=24, pady=16)

        def on_add() -> None:
            lines = [u.strip() for u in box.get("1.0", "end").splitlines() if u.strip()]
            dialog.grab_release()
            dialog.destroy()
            valid = [u for u in lines if URL_PATTERN.match(u)]
            invalid = [u for u in lines if not URL_PATTERN.match(u)]
            if invalid:
                show_warning(self.root, "Some lines were ignored",
                             "These lines do not look like URLs:\n\n"
                             + "\n".join(f"    • {u}" for u in invalid[:10]))
            if not valid:
                return
            self._maybe_expand_playlists(valid)

        def on_cancel() -> None:
            dialog.grab_release()
            dialog.destroy()

        self._button(actions, "Cancel", on_cancel).pack(side="right")
        self._button(actions, "Add to List", on_add, primary=True).pack(
            side="right", padx=8)
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        dialog.bind("<Escape>", lambda _e: on_cancel())

        dialog.wait_visibility()
        dialog.grab_set()

    def _maybe_expand_playlists(self, urls: List[str]) -> None:
        playlists = [u for u in urls if is_expandable_playlist(u)]
        if not playlists:
            self._add_tasks(urls)
            return
        expand = ask_yes_no(
            self.root, "Link points into a playlist",
            f"{len(playlists)} of {len(urls)} link(s) point to a video that sits "
            "inside a playlist.\n\n"
            "ADD ALL — queue every video in those playlists. That can be dozens "
            "of videos, each producing its own transcript.\n\n"
            "JUST THIS — queue only the linked video and ignore the rest of the "
            "playlist.\n\nWhen in doubt, choose JUST THIS.",
            yes="Add All", no="Just This")
        if not expand:
            self._add_tasks(urls)
            return
        # Reading playlists hits the network, so it runs off the main thread.
        # Controls stay locked until it finishes — otherwise the user could start
        # a batch that silently excludes the videos still being resolved.
        self.is_expanding = True
        self._set_controls_enabled(False)
        self._set_status("Reading playlist(s)…")
        threading.Thread(target=self._expand_playlists_task,
                         args=(urls, playlists), daemon=True).start()

    def _expand_playlists_task(self, all_urls: List[str], playlists: List[str]) -> None:
        resolved: List[str] = []
        for url in all_urls:
            if url not in playlists:
                resolved.append(url)
                continue
            try:
                resolved.extend(expand_playlist(url) or [url])
            except Exception as exc:
                self.event_queue.put(
                    ("status", f"Could not read playlist: {friendly_download_error(exc)}"))
                resolved.append(url)
        self.event_queue.put(("expand_done", resolved))

    # -- output folder -------------------------------------------------------

    def _select_output_directory(self) -> None:
        path = filedialog.askdirectory(title="Select the output folder")
        if path:
            self.output_directory = Path(path)
            self.output_var.set(str(self.output_directory))

    def _open_output_dir(self) -> None:
        if not self.output_directory.is_dir():
            show_info(self.root, "No folder", "No output folder has been selected yet.")
            return
        if sys.platform == "win32":
            os.startfile(self.output_directory)  # noqa: S606 - Windows explorer
        else:
            opener = "xdg-open" if sys.platform.startswith("linux") else "open"
            subprocess.Popen([opener, str(self.output_directory)])

    # -- VRAM estimate -------------------------------------------------------

    def _effective_compute_type(self) -> str:
        chosen = self.precision_var.get()
        if chosen != AUTO_PRECISION:
            return chosen
        return auto_compute_type(self.model_var.get(), self.device_info)

    def _refresh_estimate(self) -> None:
        if not self.probe_complete:
            return
        if self.backend_var.get() != FasterWhisperBackend.name:
            self.estimate_label.configure(text="openai-whisper uses ~2x more memory",
                                          text_color=color("warning"))
            return
        compute = self._effective_compute_type()
        if not self.device_info.is_cuda:
            self.estimate_label.configure(text=f"CPU · {compute}",
                                          text_color=color("text_muted"))
            return
        needed = vram_estimate_mb(self.model_var.get(), compute)
        if not self.device_info.vram_known:
            self.estimate_label.configure(
                text=f"~{needed} MiB · free VRAM unknown", text_color=color("warning"))
            return
        budget = self.device_info.usable_vram_mb
        if needed > budget:
            self.estimate_label.configure(
                text=f"⚠  ~{needed} MiB > {budget} MiB free ({compute})",
                text_color=color("danger"))
        else:
            self.estimate_label.configure(
                text=f"✓  ~{needed} MiB VRAM ({compute})", text_color=color("success"))

    # -- processing ----------------------------------------------------------

    def _selected_formats(self) -> List[str]:
        return [fmt for fmt, var in self.format_vars.items() if var.get()]

    def _start_processing(self) -> None:
        if self.is_processing or self.is_expanding:
            return

        # Only tasks that have not already produced output are re-run. Without
        # this, going Back from the results view and pressing Transcribe again
        # re-downloads and re-transcribes everything that already succeeded.
        pending = [tid for tid in self.task_order if not self.tasks[tid].is_done]
        if not self.task_order:
            show_warning(self.root, "Nothing to do",
                         "Add media files or URLs to the source list first.")
            return
        if not pending:
            show_info(self.root, "Everything is done",
                      "Every source in the list has already been transcribed.\n\n"
                      "Use 'Clear Completed' to start a new batch.")
            return
        if not self._selected_formats():
            show_warning(self.root, "No output format",
                         "Select at least one output format (.txt, .srt, .vtt, .json).")
            return

        tasks = [self.tasks[tid] for tid in pending]
        needs_folder = len(tasks) > 1 or any(task.is_url for task in tasks)
        if needs_folder and not self.output_directory.is_dir():
            show_error(self.root, "Output folder required",
                       "Processing multiple sources or URLs writes each transcript "
                       "to disk, so an output folder must be selected.")
            return

        if any(task.is_url for task in tasks) and not self.js_runtime.ok:
            if not ask_yes_no(
                self.root, "JavaScript runtime missing",
                f"{self.js_runtime.problem}\n\n{JS_RUNTIME_HELP}\n\n"
                "URL downloads will most likely fail. Continue anyway?",
                yes="Continue", no="Go Back"):
                return

        # The legacy backend shells out to the FFmpeg binary; faster-whisper does not.
        if self.backend_var.get() == OpenAIWhisperBackend.name and not self.ffmpeg_path.is_file():
            show_error(self.root, "FFmpeg required",
                       "The 'openai-whisper' backend needs FFmpeg installed and on "
                       "the PATH.\n\nUse the default 'faster-whisper' backend, which "
                       "decodes through PyAV and needs no FFmpeg.")
            return

        if (self.device_info.is_cuda
                and self.backend_var.get() == FasterWhisperBackend.name
                and self.device_info.vram_known):
            compute = self._effective_compute_type()
            needed = vram_estimate_mb(self.model_var.get(), compute)
            budget = self.device_info.usable_vram_mb
            if needed > budget and not ask_yes_no(
                self.root, "Not enough VRAM",
                f"Model '{self.model_var.get()}' at {compute} needs about "
                f"{needed} MiB, but only ~{budget} MiB are free.\n\n"
                "Transcription will spill into system memory and become VERY slow.\n\n"
                "Suggestion: use 'large-v3-turbo' with 'int8_float16' precision.\n\n"
                "Continue anyway?",
                yes="Continue", no="Go Back"):
                return

        self._persist_settings()
        self.is_processing = True
        self.cancel_event.clear()
        self._set_controls_enabled(False)
        self.cancel_button.configure(state="normal")
        self.total_bar.set(0)
        self.task_bar.set(0)
        threading.Thread(target=self._process_batch, args=(pending,), daemon=True).start()

    def _cancel_processing(self) -> None:
        if self.is_processing:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self._set_status("Cancelling — the current task will stop shortly…")

    def _process_batch(self, task_ids: List[str]) -> None:
        temp_root = Path(tempfile.gettempdir()) / "localscribe"
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            # Only build a backend when the selection actually changed; the old
            # code allocated one on every run purely to compare its type.
            wanted = self.backend_var.get()
            if wanted != self.backend.name:
                self.backend.unload()
                self.backend = BACKENDS[wanted]()

            model_name = self.model_var.get()
            compute_type = self._effective_compute_type()
            language = self.LANGUAGES[self.language_var.get()]
            task_type = "translate" if self.translate_var.get() else "transcribe"
            use_vad = self.vad_var.get()
            formats = self._selected_formats()
            save_to_disk = self.output_directory.is_dir()

            self.event_queue.put(("status", f"Loading model '{model_name}' "
                                            f"({self.device_info.device}/{compute_type})…"))
            self.backend.load(model_name, self.device_info.device, compute_type)

            claimed: set = set()  # output paths already taken by this batch
            total = len(task_ids)
            for index, task_id in enumerate(task_ids):
                if self.cancel_event.is_set():
                    self._mark_remaining_cancelled(task_ids[index:])
                    break
                self.event_queue.put(("total_progress", index / total))
                self.event_queue.put(("task_progress", 0.0))
                try:
                    self._process_one(task_id, self.tasks[task_id], temp_root, language,
                                      task_type, use_vad, formats, save_to_disk, claimed)
                except Cancelled:
                    self._mark_remaining_cancelled(task_ids[index:])
                    break

            self.event_queue.put(("total_progress", 1.0))
            self.event_queue.put(("results_ready", None))
        except Exception as exc:
            self.event_queue.put(("error", f"Batch processing failed: {exc}"))
            traceback.print_exc()
        finally:
            self.event_queue.put(("finish", None))

    def _process_one(self, task_id: str, task: Task, temp_root: Path,
                     language: Optional[str], task_type: str, use_vad: bool,
                     formats: List[str], save_to_disk: bool, claimed: set) -> None:
        # Each URL task downloads into its own directory, which is removed whole
        # afterwards. yt-dlp leaves .part and fragment files behind on a failed
        # or cancelled download, and those are not covered by deleting only the
        # final file — which is all the previous version tracked.
        work_dir: Optional[Path] = None
        media_path: Optional[Path] = None
        output_stem = Path(str(task.source)).stem

        try:
            if task.is_url:
                work_dir = temp_root / task_id
                work_dir.mkdir(parents=True, exist_ok=True)
                self._update_task(task_id, Status.WORKING, "Downloading…")
                try:
                    media_path, info = download_media(
                        task.source, work_dir, self._download_hook, self.cancel_event,
                        lambda n, wait: self._update_task(
                            task_id, Status.WORKING,
                            f"Retry {n}/{DOWNLOAD_ATTEMPTS} in {wait}s"),
                    )
                    output_stem = safe_output_stem(info.get("title", ""), media_path.stem)
                except Exception as exc:
                    # The progress hook raises Cancelled, which yt-dlp re-wraps.
                    if self.cancel_event.is_set():
                        raise Cancelled() from exc
                    self._fail_task(task_id, task, "Download failed",
                                    friendly_download_error(exc))
                    return
            else:
                media_path = Path(task.source)
                if not media_path.is_file():
                    self._fail_task(task_id, task, "File not found",
                                    f"The file '{media_path}' does not exist.")
                    return

            try:
                self._update_task(task_id, Status.WORKING, "Transcribing…")
                result = self.backend.transcribe(
                    media_path, language, task_type, use_vad,
                    lambda pct: self.event_queue.put(("task_progress", pct / 100.0)),
                    self.cancel_event,
                )
                task.text = result.text
                task.segments = result.segments
                if save_to_disk:
                    task.saved = self._write_outputs(output_stem, result, formats, claimed)
                    detail = "Saved · " + ", ".join(f".{f}" for f in formats)
                else:
                    detail = "Transcribed"
                self._update_task(task_id, Status.DONE, detail)
            except Cancelled:
                self._update_task(task_id, Status.CANCELLED, "Cancelled")
                raise
            except Exception as exc:
                self._fail_task(task_id, task, "Transcription failed", str(exc))
                traceback.print_exc()
        finally:
            if work_dir is not None:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _write_outputs(self, stem: str, result: TranscriptionResult,
                       formats: List[str], claimed: set) -> List[Path]:
        """Write every requested format, never clobbering an existing file.

        Two sources can easily share a title — think a playlist with repeated
        episode names, or identically named files from different folders. The
        suffix is resolved once for the whole set so a task's .txt and .srt keep
        matching names.
        """
        stem = self._unique_stem(stem, formats, claimed)
        written: List[Path] = []
        for fmt in formats:
            path = self.output_directory / f"{stem}.{fmt}"
            path.write_text(WRITERS[fmt](result.segments, result.text), encoding="utf-8")
            claimed.add(path)
            written.append(path)
        return written

    def _unique_stem(self, stem: str, formats: List[str], claimed: set) -> str:
        counter = 1
        candidate = stem
        while True:
            paths = [self.output_directory / f"{candidate}.{fmt}" for fmt in formats]
            if not any(p.exists() or p in claimed for p in paths):
                return candidate
            counter += 1
            candidate = f"{stem} ({counter})"

    def _update_task(self, task_id: str, status: str, detail: str) -> None:
        task = self.tasks.get(task_id)
        if task is not None:
            task.status = status
            task.detail = detail
        self.event_queue.put(("task_status", task_id))

    def _fail_task(self, task_id: str, task: Task, detail: str, error: str) -> None:
        task.error = error
        self._update_task(task_id, Status.FAILED, detail)

    def _mark_remaining_cancelled(self, task_ids: List[str]) -> None:
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task is not None and task.status in (Status.QUEUED, Status.WORKING):
                self._update_task(task_id, Status.CANCELLED, "Cancelled")

    def _download_hook(self, data: Dict[str, Any]) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            if total:
                fraction = data.get("downloaded_bytes", 0) / total
                self.event_queue.put(("task_progress", min(fraction, 1.0)))

    # -- event pump ----------------------------------------------------------

    def _pump_events(self) -> None:
        try:
            while not self.event_queue.empty():
                message, payload = self.event_queue.get_nowait()
                self._handle_event(message, payload)
        finally:
            self.root.after(80, self._pump_events)

    def _handle_event(self, message: str, payload: Any) -> None:
        if message == "status":
            self._set_status(payload)
        elif message == "task_progress":
            self.task_bar.set(max(0.0, min(float(payload), 1.0)))
        elif message == "total_progress":
            self.total_bar.set(max(0.0, min(float(payload), 1.0)))
        elif message == "task_status":
            row, task = self.rows.get(payload), self.tasks.get(payload)
            if row is not None and task is not None:
                row.set_status(task)
                self._set_status(f"{task.display[:60]} — {task.detail}")
        elif message == "probe_done":
            self._on_probe_done(payload)
        elif message == "expand_done":
            self.is_expanding = False
            self._set_controls_enabled(True)
            self._add_tasks(payload)
        elif message == "results_ready":
            self._show_results()
        elif message == "error":
            self._handle_error(payload)
        elif message == "finish":
            self.is_processing = False
            self._set_controls_enabled(True)
            self.cancel_button.configure(state="disabled")
            self.task_bar.set(0)
            self._set_status("Batch cancelled." if self.cancel_event.is_set()
                             else "Batch complete.")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        widgets = [self.add_files_button, self.add_urls_button, self.clear_button,
                   self.clear_done_button, self.model_menu, self.precision_menu,
                   self.language_menu, self.backend_menu, self.translate_switch,
                   self.vad_switch, self.browse_button, self.start_button,
                   *self.format_boxes]
        for widget in widgets:
            try:
                widget.configure(state=state)
            except (tk.TclError, ValueError):
                pass
        for row in self.rows.values():
            row.set_enabled(enabled)

    # -- results -------------------------------------------------------------

    def _show_results(self) -> None:
        for row in self.result_rows.values():
            row.destroy()
        self.result_rows.clear()

        for index, task_id in enumerate(self.task_order):
            task = self.tasks[task_id]
            row = TaskRow(self.result_list, task_id, task,
                          on_remove=lambda _tid: None, on_select=self._render_result)
            row.remove_button.grid_remove()
            row.grid(row=index, column=0, sticky="ew", padx=8,
                     pady=(6 if index == 0 else 3, 3))
            self.result_rows[task_id] = row

        if self.task_order:
            self._render_result(self.task_order[0])
        self.show_results_view()

    def _render_result(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            return
        for tid, row in self.result_rows.items():
            row.set_selected(tid == task_id)

        self.result_title.configure(text=task.display)
        body = [f"Status: {task.detail}"]
        if task.saved:
            body.append("Saved: " + ", ".join(p.name for p in task.saved))
        body.append("")
        body.append(task.error if task.error else (task.text or "(no text produced)"))

        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "\n".join(body))
        self.result_text.configure(state="disabled")
        self.selected_task = task_id

    def _save_current(self) -> None:
        task = self.tasks.get(self.selected_task) if self.selected_task else None
        if task is None or not task.text:
            show_warning(self.root, "Nothing to save",
                         "Select a completed transcription first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile=f"{Path(task.display).stem}.txt",
            filetypes=[("Text", "*.txt"), ("SubRip subtitle", "*.srt"),
                       ("WebVTT subtitle", "*.vtt"), ("JSON", "*.json"),
                       ("All files", "*.*")],
            title="Save transcription")
        if not path:
            return
        try:
            suffix = Path(path).suffix.lstrip(".").lower()
            writer = WRITERS.get(suffix, to_txt)
            Path(path).write_text(writer(task.segments, task.text), encoding="utf-8")
            self._set_status(f"Saved to {path}")
        except OSError as exc:
            show_error(self.root, "Could not save", str(exc))

    def _copy_to_clipboard(self) -> None:
        task = self.tasks.get(self.selected_task) if self.selected_task else None
        content = task.text if task and task.text else self.result_text.get("1.0", "end")
        self.root.clipboard_clear()
        self.root.clipboard_append(content.strip())
        self._set_status("Copied to clipboard.")

    # -- helpers -------------------------------------------------------------

    def _handle_error(self, message: str) -> None:
        self.is_processing = False
        self.total_bar.set(0)
        self.task_bar.set(0)
        self._set_status("An error occurred.")
        self._set_controls_enabled(True)
        show_error(self.root, "Error", str(message))

    def _set_status(self, message: str) -> None:
        self.status_label.configure(text=message)

    def _persist_settings(self) -> None:
        self.settings.update({
            "model": self.model_var.get(),
            "compute_type": self.precision_var.get(),
            "language": self.language_var.get(),
            "vad": self.vad_var.get(),
            "formats": self._selected_formats(),
            "output_dir": str(self.output_directory) if self.output_directory.is_dir() else "",
            "appearance": ctk.get_appearance_mode().lower(),
        })
        save_settings(self.settings)

    def _on_close(self) -> None:
        if self.is_processing and not ask_yes_no(
            self.root, "Transcription in progress",
            "A batch is still running. Quit anyway?",
            yes="Quit", no="Stay", destructive=True,
        ):
            return
        self.cancel_event.set()
        self._persist_settings()
        self.backend.unload()
        self.root.destroy()

    # -- FFmpeg (optional) ---------------------------------------------------
    #
    # FFmpeg is no longer required: PyAV, a faster-whisper dependency, decodes
    # the common formats. It is still located for the openai-whisper backend and
    # for exotic containers.

    def _find_ffmpeg_path(self) -> Path:
        exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        local = Path(__file__).parent.resolve() / "ffmpeg" / exe
        if local.is_file():
            return local
        system = shutil.which("ffmpeg")
        return Path(system) if system else Path()

    def _setup_ffmpeg_environment(self) -> None:
        self.ffmpeg_path = self._find_ffmpeg_path()
        if not self.ffmpeg_path.is_file():
            return
        ffmpeg_dir = str(self.ffmpeg_path.parent)
        current = os.environ.get("PATH", "")
        parts = [p for p in current.split(os.pathsep) if p]
        if not any(os.path.normcase(p) == os.path.normcase(ffmpeg_dir) for p in parts):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current


def main() -> None:
    root = DnDCTk()
    LocalScribeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
